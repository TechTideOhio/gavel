#![allow(clippy::unwrap_used)]

use gavel::classifier::{evaluate_case, ConfusionMatrix, JudgeAlignmentRegime};
use gavel::dataset::{
    load_dataset, stratified_split_dataset, validate_critique_before_verdict,
    verify_split_isolation, BinaryVerdict, DatasetError, DatasetSplit,
};
use gavel::debias::{debias_production_run, rogan_gladen_estimate};
use gavel::lockfile::{generate_lockfile, verify_lockfile};
use gavel::repo_audit::audit_repository;
use gavel::spec::SuiteSpec;
use rstest::rstest;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("packs")
        .join("fixtures")
        .join(name)
}

fn suite_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("packs")
        .join("softfactory.toml")
}

// ----------------------------------------------------------------------------
// Layer 1: Unit Tests
// ----------------------------------------------------------------------------

#[test]
fn test_confusion_matrix_metrics() {
    let mut mat = ConfusionMatrix::new();
    // 20 TP, 5 FN (25 positives, TPR = 0.80)
    for _ in 0..20 {
        mat.record(BinaryVerdict::Pass, BinaryVerdict::Pass);
    }
    for _ in 0..5 {
        mat.record(BinaryVerdict::Pass, BinaryVerdict::Fail);
    }
    // 20 TN, 5 FP (25 negatives, TNR = 0.80)
    for _ in 0..20 {
        mat.record(BinaryVerdict::Fail, BinaryVerdict::Fail);
    }
    for _ in 0..5 {
        mat.record(BinaryVerdict::Fail, BinaryVerdict::Pass);
    }

    assert_eq!(mat.total(), 50);
    assert!((mat.tpr() - 0.80).abs() < 1e-6);
    assert!((mat.tnr() - 0.80).abs() < 1e-6);
    assert!((mat.fpr() - 0.20).abs() < 1e-6);
    assert!((mat.fnr() - 0.20).abs() < 1e-6);
    assert!((mat.youden_j() - 0.60).abs() < 1e-6);
    assert!((mat.accuracy() - 0.80).abs() < 1e-6);
    assert!((mat.cohen_kappa() - 0.60).abs() < 1e-6);
    assert!((mat.mcc() - 0.60).abs() < 1e-6);
}

#[test]
fn test_rogan_gladen_exact_inversion() {
    let tpr = 0.85;
    let tnr = 0.75;
    let true_p = 0.20;

    // Forward equation: p_obs = true_p * tpr + (1 - true_p) * (1 - tnr)
    // 0.20 * 0.85 + 0.80 * 0.25 = 0.17 + 0.20 = 0.37
    let p_obs: f64 = true_p * tpr + (1.0 - true_p) * (1.0 - tnr);
    assert!((p_obs - 0.37).abs() < 1e-6);

    // Invert via Rogan-Gladen
    let recovered = rogan_gladen_estimate(p_obs, tpr, tnr).unwrap();
    assert!((recovered - true_p).abs() < 1e-6);
}

#[test]
fn test_merkle_provenance_tamper_detection() {
    let path = fixture_path("vsp_rls_judge_120.jsonl");
    let records = load_dataset(&path).unwrap();
    assert_eq!(records.len(), 120);

    // Verify hash integrity of record 10
    let prev = &records[9].record_hash;
    let computed = records[10].compute_hash(prev);
    assert_eq!(computed, records[10].record_hash);

    // Tampering with payload changes computed hash
    let mut tampered = records[10].clone();
    tampered.candidate_response = "Tampered response bypasses RLS completely".to_string();
    let tampered_hash = tampered.compute_hash(prev);
    assert_ne!(tampered_hash, records[10].record_hash);
}

// ----------------------------------------------------------------------------
// Layer 2: Parameterized Table Tests (rstest)
// ----------------------------------------------------------------------------

#[rstest]
#[case(0.95, 0.90, 0.50, 0.525, 0.50)]
#[case(0.80, 0.80, 0.10, 0.260, 0.10)]
#[case(0.90, 0.90, 0.80, 0.740, 0.80)]
fn test_rogan_gladen_table(
    #[case] tpr: f64,
    #[case] tnr: f64,
    #[case] true_prev: f64,
    #[case] expected_p_obs: f64,
    #[case] expected_recovered: f64,
) {
    let p_obs = true_prev * tpr + (1.0 - true_prev) * (1.0 - tnr);
    assert!((p_obs - expected_p_obs).abs() < 1e-3);

    let recovered = rogan_gladen_estimate(p_obs, tpr, tnr).unwrap();
    assert!((recovered - expected_recovered).abs() < 1e-3);
}

// ----------------------------------------------------------------------------
// Layer 3: Property & Invariant Tests
// ----------------------------------------------------------------------------

#[test]
fn test_stratified_split_preserves_class_balance() {
    let path = fixture_path("vsp_rls_judge_120.jsonl");
    let records = load_dataset(&path).unwrap();

    let split_records = stratified_split_dataset(records, 0.20, 0.40, 9999);
    assert_eq!(split_records.len(), 120);

    let train = split_records
        .iter()
        .filter(|r| r.split == DatasetSplit::Train)
        .count();
    let dev = split_records
        .iter()
        .filter(|r| r.split == DatasetSplit::Dev)
        .count();
    let test = split_records
        .iter()
        .filter(|r| r.split == DatasetSplit::Test)
        .count();

    assert_eq!(train, 24);
    assert_eq!(dev, 48);
    assert_eq!(test, 48);

    // Verify hash chain re-established
    let mut prev = "genesis".to_string();
    for r in &split_records {
        assert_eq!(r.prev_hash, prev);
        let computed = r.compute_hash(&prev);
        assert_eq!(r.record_hash, computed);
        prev = r.record_hash.clone();
    }
}

#[test]
fn test_bootstrap_ci_bounds() {
    let mat = ConfusionMatrix {
        tp: 23,
        fn_count: 1,
        tn: 22,
        fp: 2,
    };
    let res = debias_production_run(&mat, 150, 500, 500, 42).unwrap();

    assert!(res.bootstrap_ci_95.0 <= res.rogan_gladen_prevalence);
    assert!(res.rogan_gladen_prevalence <= res.bootstrap_ci_95.1);
    assert!(res.bootstrap_ci_95.0 >= 0.0);
    assert!(res.bootstrap_ci_95.1 <= 1.0);
}

// ----------------------------------------------------------------------------
// Layer 4: Counterfactual Negative Fixture Tests (Fail-Closed)
// ----------------------------------------------------------------------------

#[test]
fn test_negative_fixture_verdict_before_critique_fails_closed() {
    let path = fixture_path("negative_verdict_before_critique.jsonl");
    let records = load_dataset(&path).unwrap();

    let mut found_error = false;
    for r in &records {
        if let Err(DatasetError::CritiqueOrderViolation { .. }) =
            validate_critique_before_verdict(r, 8)
        {
            found_error = true;
            break;
        }
    }
    assert!(
        found_error,
        "expected CritiqueOrderViolation was not raised"
    );
}

#[test]
fn test_negative_fixture_split_leakage_fails_closed() {
    let path = fixture_path("negative_split_leakage.jsonl");
    let records = load_dataset(&path).unwrap();

    let few_shots = vec!["vsp-rls-001".to_string()];
    let res = verify_split_isolation(&records, &few_shots, 100);
    assert!(
        matches!(
            res,
            Err(DatasetError::PromptFewShotLeakage { .. } | DatasetError::SplitContamination { .. })
        ),
        "expected leakage error, got {res:?}"
    );
}

#[test]
fn test_negative_fixture_low_tnr_sycophant_fails_closed() {
    let suite = SuiteSpec::from_file(suite_path()).unwrap();
    let case = &suite.cases[0];

    let path = fixture_path("negative_low_tnr_sycophant.jsonl");
    let records = load_dataset(&path).unwrap();

    let report = evaluate_case(case, &records);
    assert!(!report.passed);
    assert_eq!(report.regime, JudgeAlignmentRegime::SpecificityDeficit);
}

#[test]
fn test_negative_fixture_under_100_examples_fails_closed() {
    let path = fixture_path("negative_under_100_examples.jsonl");
    let records = load_dataset(&path).unwrap();

    let few_shots = vec!["vsp-rls-001".to_string()];
    let res = verify_split_isolation(&records, &few_shots, 100);
    assert!(
        matches!(
            res,
            Err(DatasetError::InsufficientExamples {
                count: 40,
                min: 100
            })
        ),
        "expected InsufficientExamples error, got {res:?}"
    );
}

// ----------------------------------------------------------------------------
// Layer 5: End-to-End Suite & Repo Hygiene Audit Self-Test
// ----------------------------------------------------------------------------

#[test]
fn test_end_to_end_softfactory_suite_passes() {
    let suite = SuiteSpec::from_file(suite_path()).unwrap();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    for case in &suite.cases {
        let p = root.join(&case.dataset_path);
        let records = load_dataset(&p).unwrap();

        verify_split_isolation(
            &records,
            &case.prompt_few_shot_ids,
            case.min_labeled_examples,
        )
        .unwrap();

        for r in &records {
            validate_critique_before_verdict(r, case.min_critique_words).unwrap();
        }

        let report = evaluate_case(case, &records);
        assert!(
            report.passed,
            "case '{}' failed: {:?}",
            case.id, report.failure_reasons
        );
        assert_eq!(report.regime, JudgeAlignmentRegime::CalibratedClassifier);
    }
}

#[test]
fn test_lockfile_generation_and_verification() {
    let suite = SuiteSpec::from_file(suite_path()).unwrap();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let lock = generate_lockfile(&suite, &root).unwrap();
    verify_lockfile(&suite, &lock, &root).unwrap();
}

#[test]
fn test_softfactory_repo_hygiene_passes_all_8_checks() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let report = audit_repository(&root);

    assert!(
        report.passed,
        "repo audit failed ({}/{} passed): {:?}",
        report.passed_checks, report.total_checks, report.checks
    );
    assert_eq!(report.passed_checks, 8);
    assert_eq!(report.total_checks, 8);
}
