use crate::dataset::{BinaryVerdict, JudgeEvaluation, LabeledJudgmentRecord};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JudgeProfile {
    CalibratedRubricV3,
    VagueRubricV1,
    LenientSycophant,
    VerdictFirstRationalizer,
}

impl JudgeProfile {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "calibrated-rubric-v3" | "calibrated" => Some(Self::CalibratedRubricV3),
            "vague-rubric-v1" | "vague" => Some(Self::VagueRubricV1),
            "lenient-sycophant" | "lenient" => Some(Self::LenientSycophant),
            "verdict-first-rationalizer" | "verdict-first" => Some(Self::VerdictFirstRationalizer),
            _ => None,
        }
    }
}

pub fn simulate_judge_evaluation(
    record: &LabeledJudgmentRecord,
    profile: JudgeProfile,
) -> JudgeEvaluation {
    let mut hasher = Sha256::new();
    hasher.update(record.example_id.as_bytes());
    let hash = hasher.finalize();
    let sample = hash[0];

    match profile {
        JudgeProfile::CalibratedRubricV3 => {
            let verdict = match record.human_label {
                BinaryVerdict::Pass => {
                    if sample < 245 {
                        BinaryVerdict::Pass
                    } else {
                        BinaryVerdict::Fail
                    }
                }
                BinaryVerdict::Fail => {
                    if sample < 235 {
                        BinaryVerdict::Fail
                    } else {
                        BinaryVerdict::Pass
                    }
                }
            };
            let critique = match verdict {
                BinaryVerdict::Pass => format!(
                    "Criterion satisfied: response thoroughly addresses requirements without omissions for {}.",
                    record.example_id
                ),
                BinaryVerdict::Fail => format!(
                    "Rubric violation: response fails tenant boundary isolation and omits mandatory criteria for {}.",
                    record.example_id
                ),
            };
            let raw_response = format!(
                r#"{{"critique": "{}", "verdict": "{}"}}"#,
                critique, verdict
            );
            JudgeEvaluation {
                critique_offset: Some(2),
                verdict_offset: Some(raw_response.find("\"verdict\"").unwrap_or(50)),
                critique,
                verdict,
                raw_response: Some(raw_response),
            }
        }
        JudgeProfile::VagueRubricV1 => {
            let verdict = match record.human_label {
                BinaryVerdict::Pass => {
                    if sample < 180 {
                        BinaryVerdict::Pass
                    } else {
                        BinaryVerdict::Fail
                    }
                }
                BinaryVerdict::Fail => {
                    if sample < 175 {
                        BinaryVerdict::Fail
                    } else {
                        BinaryVerdict::Pass
                    }
                }
            };
            let critique = format!(
                "General impressions: the candidate response looks somewhat {} overall.",
                verdict
            );
            let raw_response = format!(
                r#"{{"critique": "{}", "verdict": "{}"}}"#,
                critique, verdict
            );
            JudgeEvaluation {
                critique_offset: Some(2),
                verdict_offset: Some(raw_response.find("\"verdict\"").unwrap_or(40)),
                critique,
                verdict,
                raw_response: Some(raw_response),
            }
        }
        JudgeProfile::LenientSycophant => {
            let verdict = match record.human_label {
                BinaryVerdict::Pass => BinaryVerdict::Pass,
                BinaryVerdict::Fail => {
                    if sample < 138 {
                        BinaryVerdict::Fail
                    } else {
                        BinaryVerdict::Pass
                    }
                }
            };
            let critique = format!(
                "The response is helpful and attempts to fulfill the user request with good spirit for {}.",
                record.example_id
            );
            let raw_response = format!(
                r#"{{"critique": "{}", "verdict": "{}"}}"#,
                critique, verdict
            );
            JudgeEvaluation {
                critique_offset: Some(2),
                verdict_offset: Some(raw_response.find("\"verdict\"").unwrap_or(50)),
                critique,
                verdict,
                raw_response: Some(raw_response),
            }
        }
        JudgeProfile::VerdictFirstRationalizer => {
            let verdict = record.human_label;
            let critique = format!(
                "Rationalizing the judgment: the candidate output matches expected patterns for {}.",
                record.example_id
            );
            let raw_response = format!(
                r#"{{"verdict": "{}", "critique": "{}"}}"#,
                verdict, critique
            );
            let v_pos = raw_response.find("\"verdict\"").unwrap_or(2);
            let c_pos = raw_response.find("\"critique\"").unwrap_or(30);
            JudgeEvaluation {
                critique_offset: Some(c_pos),
                verdict_offset: Some(v_pos),
                critique,
                verdict,
                raw_response: Some(raw_response),
            }
        }
    }
}
