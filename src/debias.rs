use crate::classifier::ConfusionMatrix;
use crate::dataset::SplitMix64;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebiasedPrevalenceResult {
    pub observed_pass_rate: f64,
    pub rogan_gladen_prevalence: f64,
    pub absolute_distortion: f64,
    pub bootstrap_ci_95: (f64, f64),
    pub judge_tpr: f64,
    pub judge_tnr: f64,
    pub youden_j: f64,
    pub observed_passes: usize,
    pub total_evaluated: usize,
}

pub fn rogan_gladen_estimate(p_obs: f64, tpr: f64, tnr: f64) -> Result<f64, &'static str> {
    let j = tpr + tnr - 1.0;
    if j <= 0.001 {
        return Err("degenerate judge: Youden's J <= 0.001");
    }
    let raw = (p_obs + tnr - 1.0) / j;
    Ok(raw.clamp(0.0, 1.0))
}

pub fn debias_production_run(
    test_matrix: &ConfusionMatrix,
    observed_passes: usize,
    total_evaluated: usize,
    bootstrap_iterations: usize,
    seed: u64,
) -> Result<DebiasedPrevalenceResult, &'static str> {
    if total_evaluated == 0 {
        return Err("total evaluated cannot be zero");
    }

    let p_obs = (observed_passes as f64) / (total_evaluated as f64);
    let tpr = test_matrix.tpr();
    let tnr = test_matrix.tnr();
    let j = tpr + tnr - 1.0;

    let rg_point = rogan_gladen_estimate(p_obs, tpr, tnr)?;
    let distortion = p_obs - rg_point;

    let mut rng = SplitMix64::new(seed);
    let mut bootstrap_estimates = Vec::with_capacity(bootstrap_iterations);

    let pos_total = test_matrix.positives();
    let neg_total = test_matrix.negatives();

    if pos_total == 0 || neg_total == 0 {
        return Err("test matrix must have both positive and negative examples");
    }

    for _ in 0..bootstrap_iterations {
        let tp_star = sample_binomial(&mut rng, pos_total, tpr);
        let tn_star = sample_binomial(&mut rng, neg_total, tnr);
        let k_star = sample_binomial(&mut rng, total_evaluated, p_obs);

        let tpr_star = (tp_star as f64) / (pos_total as f64);
        let tnr_star = (tn_star as f64) / (neg_total as f64);
        let p_obs_star = (k_star as f64) / (total_evaluated as f64);

        if let Ok(est) = rogan_gladen_estimate(p_obs_star, tpr_star, tnr_star) {
            bootstrap_estimates.push(est);
        }
    }

    let ci = if bootstrap_estimates.is_empty() {
        (rg_point, rg_point)
    } else {
        bootstrap_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let low_idx = ((bootstrap_estimates.len() as f64) * 0.025).floor() as usize;
        let high_idx = (((bootstrap_estimates.len() as f64) * 0.975).ceil() as usize)
            .min(bootstrap_estimates.len() - 1);
        (bootstrap_estimates[low_idx], bootstrap_estimates[high_idx])
    };

    Ok(DebiasedPrevalenceResult {
        observed_pass_rate: p_obs,
        rogan_gladen_prevalence: rg_point,
        absolute_distortion: distortion,
        bootstrap_ci_95: ci,
        judge_tpr: tpr,
        judge_tnr: tnr,
        youden_j: j,
        observed_passes,
        total_evaluated,
    })
}

fn sample_binomial(rng: &mut SplitMix64, n: usize, p: f64) -> usize {
    if p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    let mut successes = 0;
    for _ in 0..n {
        let u = (rng.next_u64() as f64) / (u64::MAX as f64);
        if u < p {
            successes += 1;
        }
    }
    successes
}

pub fn render_reference_table(tpr: f64, tnr: f64) -> String {
    let points = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95];
    let mut out = String::new();
    out.push_str(&format!(
        "Uncalibrated Judge Mirage vs Rogan-Gladen True Prevalence (TPR={:.1}%, TNR={:.1}%)\n\n",
        tpr * 100.0,
        tnr * 100.0
    ));
    out.push_str(
        "| True Prevalence | Observed Pass Rate | Absolute Distortion | Rogan-Gladen Corrected | Diagnosis |\n",
    );
    out.push_str("|:---:|:---:|:---:|:---:|---|\n");

    for &true_p in &points {
        let p_obs = true_p * tpr + (1.0 - true_p) * (1.0 - tnr);
        let rg = rogan_gladen_estimate(p_obs, tpr, tnr).unwrap_or(p_obs);
        let dist = p_obs - rg;
        let sign = if dist >= 0.0 { "+" } else { "" };

        let diag = if dist > 0.15 {
            "Severe prevalence inflation (False Confidence Trap)"
        } else if dist > 0.05 {
            "Moderate false-positive elevation"
        } else if dist < -0.10 {
            "Severe penalty from judge fatigue (False Alarm Trap)"
        } else if dist < -0.05 {
            "Moderate false-negative depression"
        } else {
            "Balanced crossing regime"
        };

        out.push_str(&format!(
            "| **{:.1}%** | **{:.1}%** | {}{:.1}% | {:.1}% | {} |\n",
            true_p * 100.0,
            p_obs * 100.0,
            sign,
            dist * 100.0,
            rg * 100.0,
            diag
        ));
    }

    out
}
