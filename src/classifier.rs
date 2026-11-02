use crate::dataset::{BinaryVerdict, DatasetSplit, LabeledJudgmentRecord};
use crate::spec::JudgeCaseSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub tp: usize,
    pub fn_count: usize,
    pub tn: usize,
    pub fp: usize,
}

impl ConfusionMatrix {
    pub fn new() -> Self {
        Self {
            tp: 0,
            fn_count: 0,
            tn: 0,
            fp: 0,
        }
    }

    pub fn record(&mut self, human: BinaryVerdict, judge: BinaryVerdict) {
        match (human, judge) {
            (BinaryVerdict::Pass, BinaryVerdict::Pass) => self.tp += 1,
            (BinaryVerdict::Pass, BinaryVerdict::Fail) => self.fn_count += 1,
            (BinaryVerdict::Fail, BinaryVerdict::Fail) => self.tn += 1,
            (BinaryVerdict::Fail, BinaryVerdict::Pass) => self.fp += 1,
        }
    }

    pub fn total(&self) -> usize {
        self.tp + self.fn_count + self.tn + self.fp
    }

    pub fn positives(&self) -> usize {
        self.tp + self.fn_count
    }

    pub fn negatives(&self) -> usize {
        self.tn + self.fp
    }

    pub fn tpr(&self) -> f64 {
        let p = self.positives();
        if p == 0 {
            0.0
        } else {
            (self.tp as f64) / (p as f64)
        }
    }

    pub fn tnr(&self) -> f64 {
        let n = self.negatives();
        if n == 0 {
            0.0
        } else {
            (self.tn as f64) / (n as f64)
        }
    }

    pub fn fpr(&self) -> f64 {
        let n = self.negatives();
        if n == 0 {
            0.0
        } else {
            (self.fp as f64) / (n as f64)
        }
    }

    pub fn fnr(&self) -> f64 {
        let p = self.positives();
        if p == 0 {
            0.0
        } else {
            (self.fn_count as f64) / (p as f64)
        }
    }

    pub fn precision(&self) -> f64 {
        let predicted_pos = self.tp + self.fp;
        if predicted_pos == 0 {
            0.0
        } else {
            (self.tp as f64) / (predicted_pos as f64)
        }
    }

    pub fn f1(&self) -> f64 {
        let denom = 2 * self.tp + self.fp + self.fn_count;
        if denom == 0 {
            0.0
        } else {
            (2.0 * self.tp as f64) / (denom as f64)
        }
    }

    pub fn accuracy(&self) -> f64 {
        let tot = self.total();
        if tot == 0 {
            0.0
        } else {
            ((self.tp + self.tn) as f64) / (tot as f64)
        }
    }

    pub fn balanced_accuracy(&self) -> f64 {
        (self.tpr() + self.tnr()) / 2.0
    }

    pub fn youden_j(&self) -> f64 {
        self.tpr() + self.tnr() - 1.0
    }

    pub fn cohen_kappa(&self) -> f64 {
        let tot = self.total() as f64;
        if tot == 0.0 {
            return 0.0;
        }

        let po = ((self.tp + self.tn) as f64) / tot;

        let p_yes_human = ((self.tp + self.fn_count) as f64) / tot;
        let p_no_human = ((self.tn + self.fp) as f64) / tot;
        let p_yes_judge = ((self.tp + self.fp) as f64) / tot;
        let p_no_judge = ((self.tn + self.fn_count) as f64) / tot;

        let pe = (p_yes_human * p_yes_judge) + (p_no_human * p_no_judge);

        if (1.0 - pe).abs() < 1e-9 {
            if (po - 1.0).abs() < 1e-9 {
                1.0
            } else {
                0.0
            }
        } else {
            (po - pe) / (1.0 - pe)
        }
    }

    pub fn mcc(&self) -> f64 {
        let tp = self.tp as f64;
        let tn = self.tn as f64;
        let fp = self.fp as f64;
        let fn_val = self.fn_count as f64;

        let numerator = tp * tn - fp * fn_val;
        let denominator = ((tp + fp) * (tp + fn_val) * (tn + fp) * (tn + fn_val)).sqrt();

        if denominator < 1e-9 {
            0.0
        } else {
            numerator / denominator
        }
    }

    pub fn tpr_wilson_95(&self) -> (f64, f64) {
        wilson_score_interval(self.tp, self.positives(), 1.959_963_984_540_054)
    }

    pub fn tnr_wilson_95(&self) -> (f64, f64) {
        wilson_score_interval(self.tn, self.negatives(), 1.959_963_984_540_054)
    }
}

impl Default for ConfusionMatrix {
    fn default() -> Self {
        Self::new()
    }
}

fn wilson_score_interval(k: usize, n: usize, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let n_f = n as f64;
    let k_f = k as f64;
    let z2 = z * z;
    let n_tilde = n_f + z2;
    let p_tilde = (k_f + 0.5 * z2) / n_tilde;
    let margin = (z / n_tilde) * (k_f * (n_f - k_f) / n_f + 0.25 * z2).sqrt();

    ((p_tilde - margin).max(0.0), (p_tilde + margin).min(1.0))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeAlignmentRegime {
    CalibratedClassifier,
    SensitivityDeficit,
    SpecificityDeficit,
    DevOverfitDrift,
    SplitContaminationBlocked,
    CritiqueOrderBlocked,
    UnpinnedModelBlocked,
    InsufficientCorpusBlocked,
    DegenerateClassifier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitMetrics {
    pub split: DatasetSplit,
    pub matrix: ConfusionMatrix,
    pub tpr: f64,
    pub tnr: f64,
    pub fpr: f64,
    pub fnr: f64,
    pub precision: f64,
    pub f1: f64,
    pub accuracy: f64,
    pub balanced_accuracy: f64,
    pub youden_j: f64,
    pub cohen_kappa: f64,
    pub mcc: f64,
    pub tpr_wilson_95: (f64, f64),
    pub tnr_wilson_95: (f64, f64),
}

impl SplitMetrics {
    pub fn from_matrix(split: DatasetSplit, matrix: ConfusionMatrix) -> Self {
        Self {
            split,
            matrix,
            tpr: matrix.tpr(),
            tnr: matrix.tnr(),
            fpr: matrix.fpr(),
            fnr: matrix.fnr(),
            precision: matrix.precision(),
            f1: matrix.f1(),
            accuracy: matrix.accuracy(),
            balanced_accuracy: matrix.balanced_accuracy(),
            youden_j: matrix.youden_j(),
            cohen_kappa: matrix.cohen_kappa(),
            mcc: matrix.mcc(),
            tpr_wilson_95: matrix.tpr_wilson_95(),
            tnr_wilson_95: matrix.tnr_wilson_95(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseCalibrationReport {
    pub case_id: String,
    pub stratum: String,
    pub pinned_model: String,
    pub prompt_version: String,
    pub train_metrics: SplitMetrics,
    pub dev_metrics: SplitMetrics,
    pub test_metrics: SplitMetrics,
    pub dev_to_test_gap: f64,
    pub regime: JudgeAlignmentRegime,
    pub passed: bool,
    pub failure_reasons: Vec<String>,
}

pub fn evaluate_case(
    case: &JudgeCaseSpec,
    records: &[LabeledJudgmentRecord],
) -> CaseCalibrationReport {
    let mut train_mat = ConfusionMatrix::new();
    let mut dev_mat = ConfusionMatrix::new();
    let mut test_mat = ConfusionMatrix::new();

    for r in records {
        let Some(ref pred) = r.judge_prediction else {
            continue;
        };
        match r.split {
            DatasetSplit::Train => train_mat.record(r.human_label, pred.verdict),
            DatasetSplit::Dev => dev_mat.record(r.human_label, pred.verdict),
            DatasetSplit::Test => test_mat.record(r.human_label, pred.verdict),
        }
    }

    let train_metrics = SplitMetrics::from_matrix(DatasetSplit::Train, train_mat);
    let dev_metrics = SplitMetrics::from_matrix(DatasetSplit::Dev, dev_mat);
    let test_metrics = SplitMetrics::from_matrix(DatasetSplit::Test, test_mat);

    let dev_to_test_gap = dev_metrics.youden_j - test_metrics.youden_j;

    let mut failure_reasons = Vec::new();
    let mut regime = JudgeAlignmentRegime::CalibratedClassifier;

    if test_metrics.youden_j <= 0.0 {
        regime = JudgeAlignmentRegime::DegenerateClassifier;
        failure_reasons.push(format!(
            "degenerate classifier: test Youden's J is {:.3} <= 0.0",
            test_metrics.youden_j
        ));
    } else if test_metrics.tpr < case.min_test_tpr {
        regime = JudgeAlignmentRegime::SensitivityDeficit;
        failure_reasons.push(format!(
            "sensitivity deficit: test TPR is {:.3} < target {:.3}",
            test_metrics.tpr, case.min_test_tpr
        ));
    } else if test_metrics.tnr < case.min_test_tnr {
        regime = JudgeAlignmentRegime::SpecificityDeficit;
        failure_reasons.push(format!(
            "specificity deficit: test TNR is {:.3} < target {:.3}",
            test_metrics.tnr, case.min_test_tnr
        ));
    } else if test_metrics.youden_j < case.min_youden_j {
        regime = JudgeAlignmentRegime::SpecificityDeficit;
        failure_reasons.push(format!(
            "test Youden's J is {:.3} < target {:.3}",
            test_metrics.youden_j, case.min_youden_j
        ));
    } else if test_metrics.cohen_kappa < case.min_cohen_kappa {
        regime = JudgeAlignmentRegime::SpecificityDeficit;
        failure_reasons.push(format!(
            "test Cohen's kappa is {:.3} < target {:.3}",
            test_metrics.cohen_kappa, case.min_cohen_kappa
        ));
    } else if dev_to_test_gap > 0.20 && test_metrics.youden_j < 0.70 {
        regime = JudgeAlignmentRegime::DevOverfitDrift;
        failure_reasons.push(format!(
            "dev overfit drift detected: dev-to-test gap is {:.3} > 0.20",
            dev_to_test_gap
        ));
    }

    let passed = failure_reasons.is_empty();

    CaseCalibrationReport {
        case_id: case.id.clone(),
        stratum: case.stratum.clone(),
        pinned_model: case.pinned_model.clone(),
        prompt_version: case.prompt_version.clone(),
        train_metrics,
        dev_metrics,
        test_metrics,
        dev_to_test_gap,
        regime,
        passed,
        failure_reasons,
    }
}
