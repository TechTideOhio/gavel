use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

const DOMAIN_TAG_JUDGMENT: &[u8] = b"tideos-gavel-judgment-v1\0";

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("failed to open dataset file at '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse JSONL line {line} at '{path}': {source}")]
    JsonLine {
        path: String,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("dataset contains only {count} examples; minimum required is {min}")]
    InsufficientExamples { count: usize, min: usize },
    #[error("split contamination: example '{example_id}' appears in multiple splits ({splits:?})")]
    SplitContamination {
        example_id: String,
        splits: Vec<String>,
    },
    #[error("prompt few-shot leakage: example '{example_id}' is in split '{split:?}' but referenced in prompt few-shot list")]
    PromptFewShotLeakage {
        example_id: String,
        split: DatasetSplit,
    },
    #[error("critique-before-verdict violation in example '{example_id}': verdict appeared at offset {verdict_offset} before critique at {critique_offset}")]
    CritiqueOrderViolation {
        example_id: String,
        critique_offset: usize,
        verdict_offset: usize,
    },
    #[error("empty or insufficient critique in example '{example_id}': got {words} words, minimum required is {min}")]
    InsufficientCritique {
        example_id: String,
        words: usize,
        min: usize,
    },
    #[error("broken Merkle provenance chain at line {line} (id '{example_id}'): expected prev_hash '{expected}', found '{actual}'")]
    BrokenProvenanceChain {
        line: usize,
        example_id: String,
        expected: String,
        actual: String,
    },
    #[error("invalid hash at line {line} (id '{example_id}'): expected '{computed}', found '{recorded}'")]
    InvalidRecordHash {
        line: usize,
        example_id: String,
        computed: String,
        recorded: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum BinaryVerdict {
    Pass,
    Fail,
}

impl std::fmt::Display for BinaryVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail => write!(f, "FAIL"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetSplit {
    Train,
    Dev,
    Test,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeEvaluation {
    pub critique: String,
    pub verdict: BinaryVerdict,
    #[serde(default)]
    pub raw_response: Option<String>,
    #[serde(default)]
    pub critique_offset: Option<usize>,
    #[serde(default)]
    pub verdict_offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledJudgmentRecord {
    pub example_id: String,
    pub split: DatasetSplit,
    pub input_prompt: String,
    pub candidate_response: String,
    pub human_label: BinaryVerdict,
    pub annotator_id: String,
    pub human_notes: String,
    #[serde(default)]
    pub judge_prediction: Option<JudgeEvaluation>,
    #[serde(default)]
    pub prev_hash: String,
    #[serde(default)]
    pub record_hash: String,
}

impl LabeledJudgmentRecord {
    pub fn compute_hash(&self, prev_hash: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN_TAG_JUDGMENT);
        hasher.update((prev_hash.len() as u64).to_be_bytes());
        hasher.update(prev_hash.as_bytes());

        hasher.update((self.example_id.len() as u64).to_be_bytes());
        hasher.update(self.example_id.as_bytes());

        let split_tag = match self.split {
            DatasetSplit::Train => "train",
            DatasetSplit::Dev => "dev",
            DatasetSplit::Test => "test",
        };
        hasher.update((split_tag.len() as u64).to_be_bytes());
        hasher.update(split_tag.as_bytes());

        hasher.update((self.input_prompt.len() as u64).to_be_bytes());
        hasher.update(self.input_prompt.as_bytes());

        hasher.update((self.candidate_response.len() as u64).to_be_bytes());
        hasher.update(self.candidate_response.as_bytes());

        let human_tag = match self.human_label {
            BinaryVerdict::Pass => "PASS",
            BinaryVerdict::Fail => "FAIL",
        };
        hasher.update((human_tag.len() as u64).to_be_bytes());
        hasher.update(human_tag.as_bytes());

        if let Some(ref pred) = self.judge_prediction {
            let pred_tag = match pred.verdict {
                BinaryVerdict::Pass => "PASS",
                BinaryVerdict::Fail => "FAIL",
            };
            hasher.update((pred_tag.len() as u64).to_be_bytes());
            hasher.update(pred_tag.as_bytes());

            hasher.update((pred.critique.len() as u64).to_be_bytes());
            hasher.update(pred.critique.as_bytes());
        }

        format!("{:x}", hasher.finalize())
    }
}

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        if len <= 1 {
            return;
        }
        for i in (1..len).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

pub fn load_dataset(path: impl AsRef<Path>) -> Result<Vec<LabeledJudgmentRecord>, DatasetError> {
    let p = path.as_ref();
    let file = File::open(p).map_err(|e| DatasetError::Io {
        path: p.display().to_string(),
        source: e,
    })?;
    let reader = BufReader::new(file);

    let mut records = Vec::new();
    let mut expected_prev_hash = "genesis".to_string();

    for (idx, line_res) in reader.lines().enumerate() {
        let line_num = idx + 1;
        let line = line_res.map_err(|e| DatasetError::Io {
            path: p.display().to_string(),
            source: e,
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let record: LabeledJudgmentRecord =
            serde_json::from_str(trimmed).map_err(|e| DatasetError::JsonLine {
                path: p.display().to_string(),
                line: line_num,
                source: e,
            })?;

        if !record.prev_hash.is_empty() && record.prev_hash != expected_prev_hash {
            return Err(DatasetError::BrokenProvenanceChain {
                line: line_num,
                example_id: record.example_id.clone(),
                expected: expected_prev_hash,
                actual: record.prev_hash.clone(),
            });
        }

        if !record.record_hash.is_empty() {
            let computed = record.compute_hash(&expected_prev_hash);
            if computed != record.record_hash {
                return Err(DatasetError::InvalidRecordHash {
                    line: line_num,
                    example_id: record.example_id.clone(),
                    computed,
                    recorded: record.record_hash.clone(),
                });
            }
            expected_prev_hash = record.record_hash.clone();
        } else {
            expected_prev_hash = record.compute_hash(&expected_prev_hash);
        }

        records.push(record);
    }

    Ok(records)
}

pub fn verify_split_isolation(
    records: &[LabeledJudgmentRecord],
    few_shot_prompt_ids: &[String],
    min_labeled_examples: usize,
) -> Result<(), DatasetError> {
    if records.len() < min_labeled_examples {
        return Err(DatasetError::InsufficientExamples {
            count: records.len(),
            min: min_labeled_examples,
        });
    }

    let mut id_to_splits: BTreeMap<String, BTreeSet<DatasetSplit>> = BTreeMap::new();
    let few_shot_set: BTreeSet<String> = few_shot_prompt_ids.iter().cloned().collect();

    for r in records {
        id_to_splits
            .entry(r.example_id.clone())
            .or_default()
            .insert(r.split);

        if few_shot_set.contains(&r.example_id) && r.split != DatasetSplit::Train {
            return Err(DatasetError::PromptFewShotLeakage {
                example_id: r.example_id.clone(),
                split: r.split,
            });
        }
    }

    for (id, splits) in id_to_splits {
        if splits.len() > 1 {
            let split_names = splits.iter().map(|s| format!("{s:?}")).collect();
            return Err(DatasetError::SplitContamination {
                example_id: id,
                splits: split_names,
            });
        }
    }

    Ok(())
}

pub fn validate_critique_before_verdict(
    record: &LabeledJudgmentRecord,
    min_critique_words: usize,
) -> Result<(), DatasetError> {
    let Some(ref pred) = record.judge_prediction else {
        return Ok(());
    };

    let word_count = pred.critique.split_whitespace().count();
    if word_count < min_critique_words {
        return Err(DatasetError::InsufficientCritique {
            example_id: record.example_id.clone(),
            words: word_count,
            min: min_critique_words,
        });
    }

    if let (Some(c_off), Some(v_off)) = (pred.critique_offset, pred.verdict_offset) {
        if v_off <= c_off {
            return Err(DatasetError::CritiqueOrderViolation {
                example_id: record.example_id.clone(),
                critique_offset: c_off,
                verdict_offset: v_off,
            });
        }
    } else if let Some(ref raw) = pred.raw_response {
        let c_pos = raw.find("\"critique\"");
        let v_pos = raw.find("\"verdict\"");
        if let (Some(cp), Some(vp)) = (c_pos, v_pos) {
            if vp <= cp {
                return Err(DatasetError::CritiqueOrderViolation {
                    example_id: record.example_id.clone(),
                    critique_offset: cp,
                    verdict_offset: vp,
                });
            }
        }
    }

    Ok(())
}

pub fn stratified_split_dataset(
    mut records: Vec<LabeledJudgmentRecord>,
    train_ratio: f64,
    dev_ratio: f64,
    seed: u64,
) -> Vec<LabeledJudgmentRecord> {
    let mut rng = SplitMix64::new(seed);

    let mut passes = Vec::new();
    let mut fails = Vec::new();

    for r in records.drain(..) {
        match r.human_label {
            BinaryVerdict::Pass => passes.push(r),
            BinaryVerdict::Fail => fails.push(r),
        }
    }

    passes.sort_by(|a, b| a.example_id.cmp(&b.example_id));
    fails.sort_by(|a, b| a.example_id.cmp(&b.example_id));

    rng.shuffle(&mut passes);
    rng.shuffle(&mut fails);

    let partition = |mut subset: Vec<LabeledJudgmentRecord>| -> Vec<LabeledJudgmentRecord> {
        let total = subset.len();
        let train_end = ((total as f64) * train_ratio).round() as usize;
        let dev_end = (((total as f64) * (train_ratio + dev_ratio)).round() as usize).min(total);

        for (i, item) in subset.iter_mut().enumerate() {
            if i < train_end {
                item.split = DatasetSplit::Train;
            } else if i < dev_end {
                item.split = DatasetSplit::Dev;
            } else {
                item.split = DatasetSplit::Test;
            }
        }
        subset
    };

    let mut partitioned = Vec::new();
    partitioned.extend(partition(passes));
    partitioned.extend(partition(fails));

    partitioned.sort_by(|a, b| a.example_id.cmp(&b.example_id));

    let mut prev_hash = "genesis".to_string();
    for r in &mut partitioned {
        r.prev_hash = prev_hash.clone();
        r.record_hash = r.compute_hash(&prev_hash);
        prev_hash = r.record_hash.clone();
    }

    partitioned
}
