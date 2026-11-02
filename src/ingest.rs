use crate::dataset::{BinaryVerdict, DatasetSplit, LabeledJudgmentRecord};
use std::fs;
use std::path::Path;

pub fn ingest_harness_runs_as_candidates(
    runs_dir: impl AsRef<Path>,
) -> Result<Vec<LabeledJudgmentRecord>, anyhow::Error> {
    let p = runs_dir.as_ref();
    if !p.exists() {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    let entries = fs::read_dir(p)?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let session_id = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let prompt_path = path.join("prompt.md");
            let stdout_path = path.join("stdout.log");
            let result_path = path.join("result.json");

            if prompt_path.exists() && stdout_path.exists() {
                let prompt = fs::read_to_string(&prompt_path).unwrap_or_default();
                let stdout = fs::read_to_string(&stdout_path).unwrap_or_default();

                let (human_label, notes) = if result_path.exists() {
                    let content = fs::read_to_string(&result_path).unwrap_or_default();
                    if content.contains("\"status\": \"success\"") {
                        (
                            BinaryVerdict::Pass,
                            "harness execution succeeded".to_string(),
                        )
                    } else {
                        (
                            BinaryVerdict::Fail,
                            "harness reported warning or error".to_string(),
                        )
                    }
                } else {
                    (
                        BinaryVerdict::Pass,
                        "unreviewed harness session".to_string(),
                    )
                };

                records.push(LabeledJudgmentRecord {
                    example_id: session_id,
                    split: DatasetSplit::Dev,
                    input_prompt: prompt,
                    candidate_response: stdout,
                    human_label,
                    annotator_id: "agent-harness-ingest".to_string(),
                    human_notes: notes,
                    judge_prediction: None,
                    prev_hash: String::new(),
                    record_hash: String::new(),
                });
            }
        }
    }

    Ok(records)
}
