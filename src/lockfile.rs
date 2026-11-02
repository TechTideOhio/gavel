use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::spec::{JudgeCaseSpec, SuiteSpec};

const DOMAIN_TAG: &[u8] = b"tideos-gavel-lock-v1\0";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseLockEntry {
    pub case_id: String,
    pub stratum: String,
    pub dataset_path: String,
    pub dataset_sha256: String,
    pub rubric_hash: String,
    pub model_config_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GavelLockfile {
    pub schema_version: String,
    pub suite_name: String,
    pub suite_version: String,
    pub overall_lock_digest: String,
    pub cases: BTreeMap<String, CaseLockEntry>,
}

pub fn hash_file_contents(path: impl AsRef<Path>) -> Result<String, std::io::Error> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn hash_rubric(case: &JudgeCaseSpec) -> String {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_TAG);
    hasher.update(b"rubric\0");
    hasher.update(case.rubric.title.as_bytes());
    hasher.update(case.rubric.instruction.as_bytes());
    for p in &case.rubric.pass_criteria {
        hasher.update(p.as_bytes());
    }
    for f in &case.rubric.fail_criteria {
        hasher.update(f.as_bytes());
    }
    for tok in &case.rubric.required_tokens {
        hasher.update(tok.as_bytes());
    }
    for tok in &case.rubric.forbidden_tokens {
        hasher.update(tok.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub fn hash_model_config(case: &JudgeCaseSpec) -> String {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_TAG);
    hasher.update(b"model_config\0");
    hasher.update(case.pinned_model.as_bytes());
    hasher.update(&case.judge_temperature.to_be_bytes());
    hasher.update(case.prompt_version.as_bytes());
    hasher.update(&case.min_test_tpr.to_be_bytes());
    hasher.update(&case.min_test_tnr.to_be_bytes());
    hasher.update(&case.min_youden_j.to_be_bytes());
    format!("{:x}", hasher.finalize())
}

pub fn generate_lockfile(
    suite: &SuiteSpec,
    root: impl AsRef<Path>,
) -> Result<GavelLockfile, anyhow::Error> {
    let root_ref = root.as_ref();
    let mut case_entries = BTreeMap::new();
    let mut overall_hasher = Sha256::new();
    overall_hasher.update(DOMAIN_TAG);
    overall_hasher.update(suite.suite.name.as_bytes());
    overall_hasher.update(suite.suite.version.as_bytes());

    for case in &suite.cases {
        let dataset_abs = root_ref.join(&case.dataset_path);
        let dataset_sha256 = hash_file_contents(&dataset_abs)?;
        let rubric_hash = hash_rubric(case);
        let model_config_hash = hash_model_config(case);

        let entry = CaseLockEntry {
            case_id: case.id.clone(),
            stratum: case.stratum.clone(),
            dataset_path: case.dataset_path.clone(),
            dataset_sha256,
            rubric_hash,
            model_config_hash,
        };

        overall_hasher.update(&(entry.case_id.len() as u64).to_be_bytes());
        overall_hasher.update(entry.case_id.as_bytes());
        overall_hasher.update(entry.dataset_sha256.as_bytes());
        overall_hasher.update(entry.rubric_hash.as_bytes());
        overall_hasher.update(entry.model_config_hash.as_bytes());

        case_entries.insert(case.id.clone(), entry);
    }

    let overall_lock_digest = format!("{:x}", overall_hasher.finalize());

    Ok(GavelLockfile {
        schema_version: "sha256-framed-v1".to_string(),
        suite_name: suite.suite.name.clone(),
        suite_version: suite.suite.version.clone(),
        overall_lock_digest,
        cases: case_entries,
    })
}

pub fn verify_lockfile(
    suite: &SuiteSpec,
    lock: &GavelLockfile,
    root: impl AsRef<Path>,
) -> Result<(), anyhow::Error> {
    if suite.suite.name != lock.suite_name {
        anyhow::bail!(
            "lockfile suite_name '{}' does not match suite '{}'",
            lock.suite_name,
            suite.suite.name
        );
    }

    let regenerated = generate_lockfile(suite, root)?;
    if regenerated.overall_lock_digest != lock.overall_lock_digest {
        anyhow::bail!(
            "lockfile overall digest mismatch: expected '{}', found '{}'",
            lock.overall_lock_digest,
            regenerated.overall_lock_digest
        );
    }

    Ok(())
}
