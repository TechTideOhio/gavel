use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoCheckResult {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoAuditReport {
    pub target_path: String,
    pub passed_checks: usize,
    pub total_checks: usize,
    pub passed: bool,
    pub checks: Vec<RepoCheckResult>,
}

pub fn audit_repository(path: impl AsRef<Path>) -> RepoAuditReport {
    let p = path.as_ref();
    let mut checks = Vec::new();

    // 1. Required OSS Files
    let required = [
        "Cargo.toml",
        "README.md",
        "LICENSE",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "AGENTS.md",
    ];
    let mut missing = Vec::new();
    for f in &required {
        if !p.join(f).exists() {
            missing.push(*f);
        }
    }
    let pass_req = missing.is_empty();
    checks.push(RepoCheckResult {
        name: "required_files".to_string(),
        passed: pass_req,
        detail: if pass_req {
            "All 7 required OSS files present".to_string()
        } else {
            format!("Missing files: {}", missing.join(", "))
        },
    });

    // 2. Pinned Rust Toolchain
    let toolchain_p = p.join("rust-toolchain.toml");
    let pass_tc = toolchain_p.exists()
        && fs::read_to_string(&toolchain_p)
            .map(|s| s.contains("channel = \"stable\""))
            .unwrap_or(false);
    checks.push(RepoCheckResult {
        name: "pinned_toolchain".to_string(),
        passed: pass_tc,
        detail: if pass_tc {
            "rust-toolchain.toml pins stable channel".to_string()
        } else {
            "Missing or unpinned rust-toolchain.toml".to_string()
        },
    });

    // 3. Linguist Vendored Attributes
    let gitattrs = p.join(".gitattributes");
    let pass_ga = gitattrs.exists()
        && fs::read_to_string(&gitattrs)
            .map(|s| s.contains("linguist-vendored"))
            .unwrap_or(false);
    checks.push(RepoCheckResult {
        name: "linguist_vendored".to_string(),
        passed: pass_ga,
        detail: if pass_ga {
            ".gitattributes configures linguist-vendored".to_string()
        } else {
            "Missing or non-vendoring .gitattributes".to_string()
        },
    });

    // 4. CI Pull Request Trigger
    let ci_yml = p.join(".github/workflows/ci.yml");
    let pass_ci = ci_yml.exists()
        && fs::read_to_string(&ci_yml)
            .map(|s| s.contains("pull_request:"))
            .unwrap_or(false);
    checks.push(RepoCheckResult {
        name: "ci_pull_request_trigger".to_string(),
        passed: pass_ci,
        detail: if pass_ci {
            "CI workflow triggers on pull_request".to_string()
        } else {
            "Missing or misconfigured CI workflow".to_string()
        },
    });

    // 5. Pinned Action SHAs
    let release_yml = p.join(".github/workflows/release.yml");
    let mut actions_pinned = true;
    for wf in [&ci_yml, &release_yml] {
        if wf.exists() {
            if let Ok(content) = fs::read_to_string(wf) {
                for line in content.lines() {
                    if line.trim().starts_with("uses:") {
                        let parts: Vec<&str> = line.split('@').collect();
                        if parts.len() == 2 {
                            let sha = parts[1].trim();
                            if sha.len() != 40 {
                                actions_pinned = false;
                            }
                        }
                    }
                }
            }
        }
    }
    checks.push(RepoCheckResult {
        name: "pinned_action_shas".to_string(),
        passed: actions_pinned,
        detail: if actions_pinned {
            "All GitHub Actions pinned to 40-char commit SHAs".to_string()
        } else {
            "Found unpinned action refs in workflows".to_string()
        },
    });

    // 6. Typography Zero Em-Dash / En-Dash
    let mut dash_violations = Vec::new();
    scan_typography_dir(p, &mut dash_violations);
    let pass_dash = dash_violations.is_empty();
    checks.push(RepoCheckResult {
        name: "typography_zero_em_dash".to_string(),
        passed: pass_dash,
        detail: if pass_dash {
            "Zero em-dashes and en-dashes across all source and doc files".to_string()
        } else {
            format!("Found em/en dashes in: {}", dash_violations.join(", "))
        },
    });

    // 7. Secret & Credential Hygiene
    let forbidden_secrets = [".env", ".env.local", "id_rsa", "id_ed25519"];
    let mut leaked = Vec::new();
    for sec in &forbidden_secrets {
        if p.join(sec).exists() {
            leaked.push(*sec);
        }
    }
    let pass_sec = leaked.is_empty();
    checks.push(RepoCheckResult {
        name: "secret_hygiene".to_string(),
        passed: pass_sec,
        detail: if pass_sec {
            "Zero private keys or .env files detected".to_string()
        } else {
            format!("Leaked credential files: {}", leaked.join(", "))
        },
    });

    // 8. Cargo Release Profile Optimizations
    let cargo_p = p.join("Cargo.toml");
    let pass_cargo = cargo_p.exists()
        && fs::read_to_string(&cargo_p)
            .map(|s| s.contains("lto = true") && s.contains("strip = true"))
            .unwrap_or(false);
    checks.push(RepoCheckResult {
        name: "cargo_release_profile".to_string(),
        passed: pass_cargo,
        detail: if pass_cargo {
            "Cargo.toml specifies lto = true and strip = true".to_string()
        } else {
            "Missing release lto / strip optimizations in Cargo.toml".to_string()
        },
    });

    let passed_checks = checks.iter().filter(|c| c.passed).count();
    let total_checks = checks.len();

    RepoAuditReport {
        target_path: p.display().to_string(),
        passed_checks,
        total_checks,
        passed: passed_checks == total_checks,
        checks,
    }
}

fn scan_typography_dir(dir: &Path, violations: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name == "target" || name == ".git" {
                continue;
            }
            if path.is_dir() {
                scan_typography_dir(&path, violations);
            } else {
                let ext = path.extension().unwrap_or_default().to_string_lossy();
                if ["rs", "md", "toml", "json", "jsonl"].contains(&ext.as_ref()) {
                    if let Ok(content) = fs::read_to_string(&path) {
                        if content.contains('\u{2014}') || content.contains('\u{2013}') {
                            violations.push(name.to_string());
                        }
                    }
                }
            }
        }
    }
}
