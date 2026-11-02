use crate::classifier::CaseCalibrationReport;
use crate::debias::DebiasedPrevalenceResult;
use serde_json::json;

pub fn generate_mermaid_pipeline() -> String {
    let mut out = String::new();
    out.push_str("flowchart LR\n");
    out.push_str("    H[\"100+ Hand-Labeled Examples\\n(Human Ground Truth: PASS / FAIL)\"] --> S[\"Deterministic Stratified Split\\nTrain (20%) | Dev (40%) | Test (40%)\"]\n");
    out.push_str("    S --> TR[\"Train Split\\nFew-Shot Examples Only\"]\n");
    out.push_str("    S --> DV[\"Dev Split\\nPrompt & Rubric Iteration\"]\n");
    out.push_str("    S --> TS[\"Held-Out Test Vault\\nLocked Final Audit\"]\n");
    out.push_str(
        "    TR --> J[\"Pinned Binary Judge\\nWritten Rubric + Critique-Before-Verdict\"]\n",
    );
    out.push_str("    DV --> J\n");
    out.push_str("    J --> TS\n");
    out.push_str(
        "    TS --> RG[\"Rogan-Gladen Estimator\\nTrue Prevalence + 95% Bootstrap CI\"]\n",
    );
    out
}

pub fn generate_mermaid_xychart(tpr: f64, tnr: f64) -> String {
    let prevalences = [5, 10, 20, 40, 60, 80, 95];
    let mut observed_strs = Vec::new();
    let mut true_strs = Vec::new();

    for &p in &prevalences {
        let p_f = (p as f64) / 100.0;
        let p_obs = p_f * tpr + (1.0 - p_f) * (1.0 - tnr);
        observed_strs.push(format!("{:.1}", p_obs * 100.0));
        true_strs.push(format!("{:.1}", p_f * 100.0));
    }

    let mut out = String::new();
    out.push_str("xychart-beta\n");
    out.push_str("    title \"Uncalibrated Judge Observed Pass Rate vs True Capability\"\n");
    out.push_str(
        "    x-axis [\"True 5%\", \"True 10%\", \"True 20%\", \"True 40%\", \"True 60%\", \"True 80%\", \"True 95%\"]\n",
    );
    out.push_str("    y-axis \"Prevalence (%)\" 0 --> 100\n");
    out.push_str(&format!("    line [{}]\n", observed_strs.join(", ")));
    out.push_str(&format!("    line [{}]\n", true_strs.join(", ")));
    out
}

pub fn generate_opentelemetry_spans(
    reports: &[CaseCalibrationReport],
    debias: Option<&DebiasedPrevalenceResult>,
) -> String {
    let mut spans = Vec::new();

    for r in reports {
        spans.push(json!({
            "trace_id": format!("gavel-{:x}", md5_hash(&r.case_id)),
            "span_id": format!("{:x}", md5_hash(&format!("{}-test", r.case_id))),
            "name": format!("gavel.calibrate.{}", r.case_id),
            "attributes": {
                "eval.case_id": r.case_id,
                "eval.stratum": r.stratum,
                "eval.pinned_model": r.pinned_model,
                "eval.prompt_version": r.prompt_version,
                "eval.judge_tpr": r.test_metrics.tpr,
                "eval.judge_tnr": r.test_metrics.tnr,
                "eval.youden_j": r.test_metrics.youden_j,
                "eval.cohen_kappa": r.test_metrics.cohen_kappa,
                "eval.mcc": r.test_metrics.mcc,
                "eval.dev_to_test_gap": r.dev_to_test_gap,
                "eval.regime": format!("{:?}", r.regime),
                "eval.passed": r.passed,
            }
        }));
    }

    if let Some(d) = debias {
        spans.push(json!({
            "trace_id": "gavel-debias-root",
            "span_id": "gavel-debias-span",
            "name": "gavel.debias.rogan_gladen",
            "attributes": {
                "eval.observed_pass_rate": d.observed_pass_rate,
                "eval.rogan_gladen_prevalence": d.rogan_gladen_prevalence,
                "eval.absolute_distortion": d.absolute_distortion,
                "eval.bootstrap_ci_95_low": d.bootstrap_ci_95.0,
                "eval.bootstrap_ci_95_high": d.bootstrap_ci_95.1,
                "eval.judge_tpr": d.judge_tpr,
                "eval.judge_tnr": d.judge_tnr,
                "eval.youden_j": d.youden_j,
                "eval.observed_passes": d.observed_passes,
                "eval.total_evaluated": d.total_evaluated,
            }
        }));
    }

    serde_json::to_string_pretty(&spans).unwrap_or_default()
}

pub fn generate_reval_ratchet(reports: &[CaseCalibrationReport]) -> String {
    let mut case_entries = Vec::new();

    for r in reports {
        case_entries.push(json!({
            "case_id": r.case_id,
            "stratum": r.stratum,
            "passed": r.passed,
            "metrics": {
                "test_tpr": r.test_metrics.tpr,
                "test_tnr": r.test_metrics.tnr,
                "youden_j": r.test_metrics.youden_j,
                "cohen_kappa": r.test_metrics.cohen_kappa,
                "dev_to_test_gap": r.dev_to_test_gap,
            }
        }));
    }

    serde_json::to_string_pretty(&json!({
        "generator": "gavel v0.1.0",
        "format": "reval-ratchet-v1",
        "cases": case_entries,
    }))
    .unwrap_or_default()
}

fn md5_hash(s: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let res = h.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&res[0..8]);
    u64::from_be_bytes(bytes)
}
