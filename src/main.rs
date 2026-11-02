#![forbid(unsafe_code)]

use clap::{Args, Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use gavel::classifier::{evaluate_case, ConfusionMatrix};
use gavel::dataset::{
    load_dataset, stratified_split_dataset, validate_critique_before_verdict,
    verify_split_isolation, DatasetSplit,
};
use gavel::debias::{debias_production_run, render_reference_table};
use gavel::envelope::{EnvelopeStatus, HarnessEnvelope};
use gavel::ingest::ingest_harness_runs_as_candidates;
use gavel::lockfile::{generate_lockfile, verify_lockfile, GavelLockfile};
use gavel::repo_audit::audit_repository;
use gavel::report::{
    generate_mermaid_pipeline, generate_mermaid_xychart, generate_opentelemetry_spans,
    generate_reval_ratchet,
};
use gavel::spec::{CaseSeverity, SuiteSpec};

const SOFTFACTORY_PACK_STR: &str = include_str!("../packs/softfactory.toml");

#[derive(Parser, Debug)]
#[command(
    name = "gavel",
    author = "TechTide Ohio",
    version = "0.1.0",
    about = "TideOS Eval Kit (Tool 9): Align LLM-as-a-Judge like a classifier",
    long_about = None
)]
pub struct Cli {
    #[arg(long, global = true, help = "Emit canonical HarnessEnvelope<T> JSON")]
    pub json: bool,

    #[arg(
        long,
        global = true,
        default_value_t = true,
        help = "Ring terminal audio alert (\\x07) on calibration deficit, leakage, or order violation"
    )]
    pub bell: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    #[command(about = "Initialize a declarative judge evaluation suite")]
    Init(InitArgs),

    #[command(
        about = "Display the Uncalibrated Judge Mirage vs Rogan-Gladen True Prevalence Table"
    )]
    Table(TableArgs),

    #[command(about = "Partition raw labeled examples into stratified train/dev/test splits")]
    Split(SplitArgs),

    #[command(about = "Calibrate a judge against dev/test splits and check structural firewalls")]
    Calibrate(CalibrateArgs),

    #[command(
        about = "Apply Rogan-Gladen prevalence debiasing with bootstrap CI to an uncalibrated run"
    )]
    Debias(DebiasArgs),

    #[command(about = "Evaluate and gate all cases in a multi-stratum judge suite")]
    Check(CheckArgs),

    #[command(about = "Compare two judge prompt iterations head-to-head")]
    Compare(CompareArgs),

    #[command(about = "Generate a sha256-framed-v1 cryptographic lockfile")]
    Lock(LockArgs),

    #[command(about = "Verify suite evidence and digests against a cryptographic lockfile")]
    Verify(VerifyArgs),

    #[command(about = "Audit a repository for Soft Factory hygiene and zero em-dash compliance")]
    Repo(RepoArgs),

    #[command(about = "Export judge metrics as Mermaid diagrams, OTel spans, or reval ratchet")]
    Export(ExportArgs),

    #[command(about = "Ingest multi-agent harness execution runs from agent-harness/runs")]
    Ingest(IngestArgs),
}

#[derive(Args, Debug)]
pub struct InitArgs {
    #[arg(long, default_value = "softfactory", help = "Preset template pack")]
    pub pack: String,

    #[arg(
        long,
        default_value = "gavel.toml",
        help = "Output path for suite file"
    )]
    pub out: PathBuf,
}

#[derive(Args, Debug)]
pub struct TableArgs {
    #[arg(long, default_value_t = 0.85, help = "Judge sensitivity (TPR)")]
    pub tpr: f64,

    #[arg(long, default_value_t = 0.75, help = "Judge specificity (TNR)")]
    pub tnr: f64,
}

#[derive(Args, Debug)]
pub struct SplitArgs {
    #[arg(long, help = "Path to input JSONL dataset")]
    pub input: PathBuf,

    #[arg(long, help = "Path to output JSONL dataset with assigned splits")]
    pub out: PathBuf,

    #[arg(
        long,
        default_value_t = 42,
        help = "PRNG seed for deterministic SplitMix64"
    )]
    pub seed: u64,

    #[arg(long, default_value_t = 0.20, help = "Train split ratio")]
    pub train_ratio: f64,

    #[arg(long, default_value_t = 0.40, help = "Dev split ratio")]
    pub dev_ratio: f64,
}

#[derive(Args, Debug)]
pub struct CalibrateArgs {
    #[arg(long, help = "Path to suite TOML file")]
    pub suite: PathBuf,

    #[arg(long, help = "Case ID to calibrate")]
    pub case: String,

    #[arg(long, help = "Override dataset path")]
    pub dataset: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct DebiasArgs {
    #[arg(long, help = "Judge sensitivity (TPR) measured on test split")]
    pub tpr: f64,

    #[arg(long, help = "Judge specificity (TNR) measured on test split")]
    pub tnr: f64,

    #[arg(long, help = "Number of observed passes in production run")]
    pub observed_passes: usize,

    #[arg(long, help = "Total items evaluated in production run")]
    pub total: usize,

    #[arg(
        long,
        default_value_t = 23,
        help = "Test true positives for bootstrap resampling"
    )]
    pub tp: usize,

    #[arg(
        long,
        default_value_t = 2,
        help = "Test false negatives for bootstrap resampling"
    )]
    pub fn_count: usize,

    #[arg(
        long,
        default_value_t = 22,
        help = "Test true negatives for bootstrap resampling"
    )]
    pub tn: usize,

    #[arg(
        long,
        default_value_t = 2,
        help = "Test false positives for bootstrap resampling"
    )]
    pub fp: usize,

    #[arg(long, default_value_t = 2000, help = "Bootstrap iterations")]
    pub bootstrap_iterations: usize,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub seed: u64,
}

#[derive(Args, Debug)]
pub struct CheckArgs {
    #[arg(long, help = "Path to suite TOML file")]
    pub suite: PathBuf,

    #[arg(long, default_value = ".", help = "Root directory for relative paths")]
    pub root: PathBuf,

    #[arg(long, help = "Path to sha256-framed-v1 lockfile")]
    pub lock: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct CompareArgs {
    #[arg(long, help = "Path to suite TOML file")]
    pub suite: PathBuf,

    #[arg(long, help = "Case ID to compare")]
    pub case: String,

    #[arg(long, help = "Baseline dataset JSONL")]
    pub baseline: PathBuf,

    #[arg(long, help = "Candidate dataset JSONL")]
    pub candidate: PathBuf,
}

#[derive(Args, Debug)]
pub struct LockArgs {
    #[arg(long, help = "Path to suite TOML file")]
    pub suite: PathBuf,

    #[arg(long, default_value = ".", help = "Root directory for relative paths")]
    pub root: PathBuf,

    #[arg(long, help = "Output path for lockfile")]
    pub out: PathBuf,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    #[arg(long, help = "Path to suite TOML file")]
    pub suite: PathBuf,

    #[arg(long, default_value = ".", help = "Root directory for relative paths")]
    pub root: PathBuf,

    #[arg(long, help = "Path to lockfile")]
    pub lock: PathBuf,
}

#[derive(Args, Debug)]
pub struct RepoArgs {
    #[arg(default_value = ".", help = "Directory to audit")]
    pub path: PathBuf,
}

#[derive(Args, Debug)]
pub struct ExportArgs {
    #[arg(long, help = "Path to suite TOML file")]
    pub suite: PathBuf,

    #[arg(long, default_value = ".", help = "Root directory for relative paths")]
    pub root: PathBuf,

    #[arg(
        long,
        default_value = "mermaid",
        help = "Export format: mermaid | xychart | otel | reval"
    )]
    pub format: String,
}

#[derive(Args, Debug)]
pub struct IngestArgs {
    #[arg(
        long,
        default_value = "Tools/agent-harness/runs",
        help = "Harness runs directory"
    )]
    pub runs_dir: PathBuf,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let bell = cli.bell;
    let is_json = cli.json;

    let res = run_command(cli);
    match res {
        Ok(code) => {
            if code != ExitCode::SUCCESS && bell {
                eprint!("\x07");
            }
            code
        }
        Err(err) => {
            if bell {
                eprint!("\x07");
            }
            if is_json {
                let env = HarnessEnvelope::<Option<()>>::error(format!("{err:#}"), None);
                let _ = env.print_json();
            } else {
                eprintln!("Error: {err:#}");
            }
            ExitCode::from(2)
        }
    }
}

fn run_command(cli: Cli) -> Result<ExitCode, anyhow::Error> {
    match cli.command {
        Commands::Table(args) => handle_table(args, cli.json),
        Commands::Init(args) => handle_init(args, cli.json),
        Commands::Split(args) => handle_split(args, cli.json),
        Commands::Calibrate(args) => handle_calibrate(args, cli.json),
        Commands::Debias(args) => handle_debias(args, cli.json),
        Commands::Check(args) => handle_check(args, cli.json),
        Commands::Compare(args) => handle_compare(args, cli.json),
        Commands::Lock(args) => handle_lock(args, cli.json),
        Commands::Verify(args) => handle_verify(args, cli.json),
        Commands::Repo(args) => handle_repo(args, cli.json),
        Commands::Export(args) => handle_export(args, cli.json),
        Commands::Ingest(args) => handle_ingest(args, cli.json),
    }
}

fn handle_table(args: TableArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    let table = render_reference_table(args.tpr, args.tnr);
    if is_json {
        let env = HarnessEnvelope::success(
            format!(
                "Generated Rogan-Gladen prevalence distortion table for TPR={:.1}%, TNR={:.1}%",
                args.tpr * 100.0,
                args.tnr * 100.0
            ),
            serde_json::json!({
                "tpr": args.tpr,
                "tnr": args.tnr,
                "markdown_table": table,
            }),
        );
        env.print_json()?;
    } else {
        println!("{table}");
    }
    Ok(ExitCode::SUCCESS)
}

fn handle_init(args: InitArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&args.out, SOFTFACTORY_PACK_STR)?;
    if is_json {
        let env = HarnessEnvelope::success(
            format!("Initialized suite at '{}'", args.out.display()),
            serde_json::json!({ "path": args.out.display().to_string() }),
        )
        .with_artifact(args.out.display().to_string());
        env.print_json()?;
    } else {
        println!(
            "Initialized Soft Factory judge suite at '{}'",
            args.out.display()
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn handle_split(args: SplitArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    let records = load_dataset(&args.input)?;
    let partitioned =
        stratified_split_dataset(records, args.train_ratio, args.dev_ratio, args.seed);

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut lines = Vec::new();
    for r in &partitioned {
        lines.push(serde_json::to_string(r)?);
    }
    fs::write(&args.out, lines.join("\n") + "\n")?;

    let train_count = partitioned
        .iter()
        .filter(|r| r.split == DatasetSplit::Train)
        .count();
    let dev_count = partitioned
        .iter()
        .filter(|r| r.split == DatasetSplit::Dev)
        .count();
    let test_count = partitioned
        .iter()
        .filter(|r| r.split == DatasetSplit::Test)
        .count();

    if is_json {
        let env = HarnessEnvelope::success(
            format!(
                "Split {} examples into train={}, dev={}, test={}",
                partitioned.len(),
                train_count,
                dev_count,
                test_count
            ),
            serde_json::json!({
                "total": partitioned.len(),
                "train": train_count,
                "dev": dev_count,
                "test": test_count,
                "out": args.out.display().to_string(),
            }),
        )
        .with_artifact(args.out.display().to_string());
        env.print_json()?;
    } else {
        println!(
            "Partitioned {} examples (train={}, dev={}, test={}) into '{}'",
            partitioned.len(),
            train_count,
            dev_count,
            test_count,
            args.out.display()
        );
    }

    Ok(ExitCode::SUCCESS)
}

fn handle_calibrate(args: CalibrateArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    let suite = SuiteSpec::from_file(&args.suite)?;
    let case = suite
        .cases
        .iter()
        .find(|c| c.id == args.case)
        .ok_or_else(|| anyhow::anyhow!("case '{}' not found in suite", args.case))?;

    let default_dataset = PathBuf::from(&case.dataset_path);
    let dataset_path = args.dataset.as_ref().unwrap_or(&default_dataset);
    let records = load_dataset(dataset_path)?;

    // 1. Verify split isolation & minimum count
    verify_split_isolation(
        &records,
        &case.prompt_few_shot_ids,
        case.min_labeled_examples,
    )?;

    // 2. Validate critique before verdict on all records
    for r in &records {
        validate_critique_before_verdict(r, case.min_critique_words)?;
    }

    // 3. Evaluate confusion matrices and alignment metrics
    let report = evaluate_case(case, &records);

    if is_json {
        let status = if report.passed {
            EnvelopeStatus::Success
        } else {
            EnvelopeStatus::Error
        };
        let env = HarnessEnvelope {
            status,
            summary: format!(
                "Calibrate case '{}': regime={:?}, test TPR={:.1}%, TNR={:.1}%, J={:.3}",
                case.id,
                report.regime,
                report.test_metrics.tpr * 100.0,
                report.test_metrics.tnr * 100.0,
                report.test_metrics.youden_j
            ),
            next_actions: if report.passed {
                vec!["Run 'gavel check' on full suite".to_string()]
            } else {
                vec!["Tune judge prompt on dev split before re-evaluating test".to_string()]
            },
            artifacts: vec![dataset_path.display().to_string()],
            data: report.clone(),
        };
        env.print_json()?;
    } else {
        println!("========================================================");
        println!(" Case: {} ({})", case.id, case.stratum);
        println!(" Pinned Model: {}", case.pinned_model);
        println!(" Regime: {:?}", report.regime);
        println!(" Status: {}", if report.passed { "PASS" } else { "FAIL" });
        println!("--------------------------------------------------------");
        println!(
            " Dev : TPR={:.1}%, TNR={:.1}%, Youden J={:.3}, Cohen Kappa={:.3}",
            report.dev_metrics.tpr * 100.0,
            report.dev_metrics.tnr * 100.0,
            report.dev_metrics.youden_j,
            report.dev_metrics.cohen_kappa
        );
        println!(
            " Test: TPR={:.1}%, TNR={:.1}%, Youden J={:.3}, Cohen Kappa={:.3}",
            report.test_metrics.tpr * 100.0,
            report.test_metrics.tnr * 100.0,
            report.test_metrics.youden_j,
            report.test_metrics.cohen_kappa
        );
        println!(" Dev-to-Test Gap: {:.3}", report.dev_to_test_gap);
        if !report.passed {
            println!(" Failures:");
            for f in &report.failure_reasons {
                println!("   - {f}");
            }
        }
        println!("========================================================");
    }

    if report.passed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::from(1))
    }
}

fn handle_debias(args: DebiasArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    let mat = ConfusionMatrix {
        tp: args.tp,
        fn_count: args.fn_count,
        tn: args.tn,
        fp: args.fp,
    };

    let result = debias_production_run(
        &mat,
        args.observed_passes,
        args.total,
        args.bootstrap_iterations,
        args.seed,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    if is_json {
        let env = HarnessEnvelope::success(
            format!(
                "Rogan-Gladen debiasing: observed={:.1}%, true prevalence={:.1}% [95% CI: {:.1}% - {:.1}%]",
                result.observed_pass_rate * 100.0,
                result.rogan_gladen_prevalence * 100.0,
                result.bootstrap_ci_95.0 * 100.0,
                result.bootstrap_ci_95.1 * 100.0
            ),
            result,
        );
        env.print_json()?;
    } else {
        println!("========================================================");
        println!(" Rogan-Gladen Prevalence Bias Correction (1978)");
        println!("--------------------------------------------------------");
        println!(
            " Judge Accuracy   : TPR={:.1}%, TNR={:.1}%, Youden J={:.3}",
            result.judge_tpr * 100.0,
            result.judge_tnr * 100.0,
            result.youden_j
        );
        println!(
            " Observed Passes  : {} / {} ({:.1}%)",
            result.observed_passes,
            result.total_evaluated,
            result.observed_pass_rate * 100.0
        );
        println!(
            " Corrected Pass % : {:.1}%",
            result.rogan_gladen_prevalence * 100.0
        );
        println!(
            " Absolute Bias    : {:+.1}%",
            result.absolute_distortion * 100.0
        );
        println!(
            " 95% Bootstrap CI : [{:.1}% - {:.1}%]",
            result.bootstrap_ci_95.0 * 100.0,
            result.bootstrap_ci_95.1 * 100.0
        );
        println!("========================================================");
    }

    Ok(ExitCode::SUCCESS)
}

fn handle_check(args: CheckArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    let suite = SuiteSpec::from_file(&args.suite)?;

    if let Some(ref lock_path) = args.lock {
        let lock_contents = fs::read_to_string(lock_path)?;
        let lock: GavelLockfile = serde_json::from_str(&lock_contents)?;
        verify_lockfile(&suite, &lock, &args.root)?;
    }

    let mut reports = Vec::new();
    let mut blocker_failed = false;

    for case in &suite.cases {
        let dataset_path = args.root.join(&case.dataset_path);
        let records = load_dataset(&dataset_path)?;

        verify_split_isolation(
            &records,
            &case.prompt_few_shot_ids,
            case.min_labeled_examples,
        )?;

        for r in &records {
            validate_critique_before_verdict(r, case.min_critique_words)?;
        }

        let rep = evaluate_case(case, &records);
        if !rep.passed && case.severity == CaseSeverity::Blocker {
            blocker_failed = true;
        }
        reports.push(rep);
    }

    let all_passed = reports.iter().all(|r| r.passed);

    if is_json {
        let status = if all_passed {
            EnvelopeStatus::Success
        } else if blocker_failed {
            EnvelopeStatus::Error
        } else {
            EnvelopeStatus::Warning
        };

        let env = HarnessEnvelope {
            status,
            summary: format!(
                "Suite '{}': {}/{} cases passed, blocker_failed={}",
                suite.suite.name,
                reports.iter().filter(|r| r.passed).count(),
                reports.len(),
                blocker_failed
            ),
            next_actions: if all_passed {
                vec!["All judge models calibrated and verified".to_string()]
            } else {
                vec!["Inspect failing cases and tune rubrics".to_string()]
            },
            artifacts: vec![args.suite.display().to_string()],
            data: serde_json::json!({
                "suite": suite.suite.name,
                "all_passed": all_passed,
                "blocker_failed": blocker_failed,
                "cases": reports,
            }),
        };
        env.print_json()?;
    } else {
        println!("========================================================");
        println!(" Suite Check: {}", suite.suite.name);
        println!("--------------------------------------------------------");
        for r in &reports {
            println!(
                " Case {:<32} [{}] : {}",
                r.case_id,
                r.stratum,
                if r.passed { "PASS" } else { "FAIL" }
            );
        }
        println!("--------------------------------------------------------");
        println!(
            " Result: {} (Blockers Failed: {})",
            if all_passed { "PASS" } else { "FAIL" },
            blocker_failed
        );
        println!("========================================================");
    }

    if all_passed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::from(1))
    }
}

fn handle_compare(args: CompareArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    let suite = SuiteSpec::from_file(&args.suite)?;
    let case = suite
        .cases
        .iter()
        .find(|c| c.id == args.case)
        .ok_or_else(|| anyhow::anyhow!("case '{}' not found in suite", args.case))?;

    let base_records = load_dataset(&args.baseline)?;
    let cand_records = load_dataset(&args.candidate)?;

    let base_report = evaluate_case(case, &base_records);
    let cand_report = evaluate_case(case, &cand_records);

    let delta_tpr = cand_report.test_metrics.tpr - base_report.test_metrics.tpr;
    let delta_tnr = cand_report.test_metrics.tnr - base_report.test_metrics.tnr;
    let delta_j = cand_report.test_metrics.youden_j - base_report.test_metrics.youden_j;

    if is_json {
        let env = HarnessEnvelope::success(
            format!(
                "Compared case '{}': delta TPR={:+.1}%, delta TNR={:+.1}%, delta J={:+.3}",
                case.id,
                delta_tpr * 100.0,
                delta_tnr * 100.0,
                delta_j
            ),
            serde_json::json!({
                "case_id": case.id,
                "baseline": base_report,
                "candidate": cand_report,
                "delta": {
                    "tpr": delta_tpr,
                    "tnr": delta_tnr,
                    "youden_j": delta_j,
                }
            }),
        );
        env.print_json()?;
    } else {
        println!("========================================================");
        println!(" Head-to-Head Compare: {}", case.id);
        println!(" Baseline : {}", args.baseline.display());
        println!(" Candidate: {}", args.candidate.display());
        println!("--------------------------------------------------------");
        println!(
            " Baseline  Test: TPR={:.1}%, TNR={:.1}%, J={:.3}",
            base_report.test_metrics.tpr * 100.0,
            base_report.test_metrics.tnr * 100.0,
            base_report.test_metrics.youden_j
        );
        println!(
            " Candidate Test: TPR={:.1}%, TNR={:.1}%, J={:.3}",
            cand_report.test_metrics.tpr * 100.0,
            cand_report.test_metrics.tnr * 100.0,
            cand_report.test_metrics.youden_j
        );
        println!(
            " Delta          : TPR={:+.1}%, TNR={:+.1}%, J={:+.3}",
            delta_tpr * 100.0,
            delta_tnr * 100.0,
            delta_j
        );
        println!("========================================================");
    }

    Ok(ExitCode::SUCCESS)
}

fn handle_lock(args: LockArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    let suite = SuiteSpec::from_file(&args.suite)?;
    let lock = generate_lockfile(&suite, &args.root)?;
    let serialized = serde_json::to_string_pretty(&lock)?;

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&args.out, serialized)?;

    if is_json {
        let env = HarnessEnvelope::success(
            format!(
                "Generated lockfile with digest '{}'",
                lock.overall_lock_digest
            ),
            lock,
        )
        .with_artifact(args.out.display().to_string());
        env.print_json()?;
    } else {
        println!(
            "Wrote sha256-framed-v1 lockfile to '{}' (digest: {})",
            args.out.display(),
            lock.overall_lock_digest
        );
    }

    Ok(ExitCode::SUCCESS)
}

fn handle_verify(args: VerifyArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    let suite = SuiteSpec::from_file(&args.suite)?;
    let lock_contents = fs::read_to_string(&args.lock)?;
    let lock: GavelLockfile = serde_json::from_str(&lock_contents)?;

    verify_lockfile(&suite, &lock, &args.root)?;

    if is_json {
        let env = HarnessEnvelope::success(
            format!(
                "Verified lockfile at '{}' with zero drift",
                args.lock.display()
            ),
            serde_json::json!({
                "lockfile": args.lock.display().to_string(),
                "digest": lock.overall_lock_digest,
            }),
        );
        env.print_json()?;
    } else {
        println!(
            "Lockfile '{}' verified: zero drift detected.",
            args.lock.display()
        );
    }

    Ok(ExitCode::SUCCESS)
}

fn handle_repo(args: RepoArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    let report = audit_repository(&args.path);

    if is_json {
        let status = if report.passed {
            EnvelopeStatus::Success
        } else {
            EnvelopeStatus::Error
        };
        let env = HarnessEnvelope {
            status,
            summary: format!(
                "Repository audit: {}/{} checks passed",
                report.passed_checks, report.total_checks
            ),
            next_actions: if report.passed {
                vec!["All repository hygiene gates passed".to_string()]
            } else {
                vec!["Fix failing hygiene checks".to_string()]
            },
            artifacts: vec![args.path.display().to_string()],
            data: report.clone(),
        };
        env.print_json()?;
    } else {
        println!("========================================================");
        println!(" Soft Factory Repository Hygiene Audit");
        println!(" Target: {}", report.target_path);
        println!("--------------------------------------------------------");
        for c in &report.checks {
            println!(
                " [{}] {:<28} : {}",
                if c.passed { "PASS" } else { "FAIL" },
                c.name,
                c.detail
            );
        }
        println!("--------------------------------------------------------");
        println!(
            " Result: {}/{} Checks Passed ({})",
            report.passed_checks,
            report.total_checks,
            if report.passed { "PASS" } else { "FAIL" }
        );
        println!("========================================================");
    }

    if report.passed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::from(1))
    }
}

fn handle_export(args: ExportArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    let suite = SuiteSpec::from_file(&args.suite)?;

    let mut reports = Vec::new();
    for case in &suite.cases {
        let p = args.root.join(&case.dataset_path);
        if let Ok(records) = load_dataset(&p) {
            reports.push(evaluate_case(case, &records));
        }
    }

    let output = match args.format.as_str() {
        "mermaid" => generate_mermaid_pipeline(),
        "xychart" => generate_mermaid_xychart(0.85, 0.75),
        "otel" => generate_opentelemetry_spans(&reports, None),
        "reval" => generate_reval_ratchet(&reports),
        other => anyhow::bail!("unknown export format '{other}'"),
    };

    if is_json {
        let env = HarnessEnvelope::success(
            format!("Exported {} format", args.format),
            serde_json::json!({ "format": args.format, "output": output }),
        );
        env.print_json()?;
    } else {
        println!("{output}");
    }

    Ok(ExitCode::SUCCESS)
}

fn handle_ingest(args: IngestArgs, is_json: bool) -> Result<ExitCode, anyhow::Error> {
    let records = ingest_harness_runs_as_candidates(&args.runs_dir)?;
    if is_json {
        let env = HarnessEnvelope::success(
            format!("Ingested {} harness sessions", records.len()),
            serde_json::json!({
                "count": records.len(),
                "records": records,
            }),
        );
        env.print_json()?;
    } else {
        println!(
            "Ingested {} harness execution sessions from '{}'",
            records.len(),
            args.runs_dir.display()
        );
    }
    Ok(ExitCode::SUCCESS)
}
