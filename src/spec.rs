use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpecError {
    #[error("failed to read suite specification at '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse TOML specification at '{path}': {source}")]
    ParseToml {
        path: String,
        #[source]
        source: toml::de::Error,
    },
    #[error("invalid stratum weights: sum is {sum:.4}, expected 1.0")]
    InvalidWeights { sum: f64 },
    #[error("case '{case_id}' references unknown stratum '{stratum}'")]
    UnknownStratum { case_id: String, stratum: String },
    #[error("case '{case_id}' requires at least 100 labeled examples, specified {count}")]
    InsufficientMinimumExamples { case_id: String, count: usize },
    #[error("case '{case_id}' specifies non-zero judge temperature {temp:.4}; judge decoding must be pinned to 0.0")]
    NonZeroJudgeTemperature { case_id: String, temp: f64 },
    #[error("case '{case_id}' uses floating model alias '{model}'; judge model must be pinned to exact snapshot or SHA digest")]
    UnpinnedJudgeModel { case_id: String, model: String },
    #[error("case '{case_id}' split ratios sum to {sum:.4}, expected 1.0")]
    InvalidSplitRatios { case_id: String, sum: f64 },
    #[error("case '{case_id}' has empty pass or fail criteria in rubric")]
    EmptyRubricCriteria { case_id: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaseSeverity {
    Blocker,
    Graded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteMeta {
    pub name: String,
    pub version: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrataWeights {
    pub critical: f64,
    pub core: f64,
    pub hygiene: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RubricSpec {
    pub title: String,
    pub instruction: String,
    pub pass_criteria: Vec<String>,
    pub fail_criteria: Vec<String>,
    #[serde(default)]
    pub required_tokens: Vec<String>,
    #[serde(default)]
    pub forbidden_tokens: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitPolicySpec {
    #[serde(default = "default_train_ratio")]
    pub train_ratio: f64,
    #[serde(default = "default_dev_ratio")]
    pub dev_ratio: f64,
    #[serde(default = "default_test_ratio")]
    pub test_ratio: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_train_ratio() -> f64 {
    0.20
}

fn default_dev_ratio() -> f64 {
    0.40
}

fn default_test_ratio() -> f64 {
    0.40
}

fn default_seed() -> u64 {
    42
}

impl Default for SplitPolicySpec {
    fn default() -> Self {
        Self {
            train_ratio: default_train_ratio(),
            dev_ratio: default_dev_ratio(),
            test_ratio: default_test_ratio(),
            seed: default_seed(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeCaseSpec {
    pub id: String,
    pub stratum: String,
    pub severity: CaseSeverity,
    pub description: String,
    pub dataset_path: String,
    pub pinned_model: String,
    #[serde(default)]
    pub judge_temperature: f64,
    pub prompt_version: String,
    pub min_labeled_examples: usize,
    pub min_test_tpr: f64,
    pub min_test_tnr: f64,
    pub min_youden_j: f64,
    #[serde(default = "default_min_cohen_kappa")]
    pub min_cohen_kappa: f64,
    #[serde(default = "default_min_critique_words")]
    pub min_critique_words: usize,
    pub rubric: RubricSpec,
    #[serde(default)]
    pub split_policy: SplitPolicySpec,
    #[serde(default)]
    pub prompt_few_shot_ids: Vec<String>,
}

fn default_min_cohen_kappa() -> f64 {
    0.60
}

fn default_min_critique_words() -> usize {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteSpec {
    pub suite: SuiteMeta,
    pub weights: StrataWeights,
    pub cases: Vec<JudgeCaseSpec>,
}

impl SuiteSpec {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, SpecError> {
        let p = path.as_ref();
        let contents = fs::read_to_string(p).map_err(|e| SpecError::Io {
            path: p.display().to_string(),
            source: e,
        })?;
        let spec: Self = toml::from_str(&contents).map_err(|e| SpecError::ParseToml {
            path: p.display().to_string(),
            source: e,
        })?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<(), SpecError> {
        let sum = self.weights.critical + self.weights.core + self.weights.hygiene;
        if (sum - 1.0).abs() > 0.001 {
            return Err(SpecError::InvalidWeights { sum });
        }

        let pinned_model_regex = Regex::new(
            r"(?i)^([a-z0-9._-]+-(20\d{2}-\d{2}-\d{2}|20\d{6})|[a-z0-9._-]+@sha256:[a-f0-9]{64})$",
        )
        .map_err(|e| SpecError::Io {
            path: "regex".to_string(),
            source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
        })?;

        for case in &self.cases {
            let valid_stratum = matches!(case.stratum.as_str(), "critical" | "core" | "hygiene");
            if !valid_stratum {
                return Err(SpecError::UnknownStratum {
                    case_id: case.id.clone(),
                    stratum: case.stratum.clone(),
                });
            }

            if case.min_labeled_examples < 100 {
                return Err(SpecError::InsufficientMinimumExamples {
                    case_id: case.id.clone(),
                    count: case.min_labeled_examples,
                });
            }

            if case.judge_temperature.abs() > 0.0001 {
                return Err(SpecError::NonZeroJudgeTemperature {
                    case_id: case.id.clone(),
                    temp: case.judge_temperature,
                });
            }

            if !pinned_model_regex.is_match(&case.pinned_model) {
                return Err(SpecError::UnpinnedJudgeModel {
                    case_id: case.id.clone(),
                    model: case.pinned_model.clone(),
                });
            }

            let split_sum = case.split_policy.train_ratio
                + case.split_policy.dev_ratio
                + case.split_policy.test_ratio;
            if (split_sum - 1.0).abs() > 0.001 {
                return Err(SpecError::InvalidSplitRatios {
                    case_id: case.id.clone(),
                    sum: split_sum,
                });
            }

            if case.rubric.pass_criteria.is_empty() || case.rubric.fail_criteria.is_empty() {
                return Err(SpecError::EmptyRubricCriteria {
                    case_id: case.id.clone(),
                });
            }
        }

        Ok(())
    }
}
