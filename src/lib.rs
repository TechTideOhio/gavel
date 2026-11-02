#![forbid(unsafe_code)]

pub mod classifier;
pub mod dataset;
pub mod debias;
pub mod economics;
pub mod envelope;
pub mod ingest;
pub mod lockfile;
pub mod repo_audit;
pub mod report;
pub mod simulator;
pub mod spec;

pub use classifier::{
    evaluate_case, CaseCalibrationReport, ConfusionMatrix, JudgeAlignmentRegime, SplitMetrics,
};
pub use dataset::{
    load_dataset, stratified_split_dataset, validate_critique_before_verdict,
    verify_split_isolation, BinaryVerdict, DatasetError, DatasetSplit, JudgeEvaluation,
    LabeledJudgmentRecord, SplitMix64,
};
pub use debias::{
    debias_production_run, render_reference_table, rogan_gladen_estimate, DebiasedPrevalenceResult,
};
pub use economics::JudgeTokenEconomics;
pub use envelope::{EnvelopeStatus, HarnessEnvelope};
pub use lockfile::{generate_lockfile, verify_lockfile, GavelLockfile};
pub use repo_audit::{audit_repository, RepoAuditReport};
pub use simulator::{simulate_judge_evaluation, JudgeProfile};
pub use spec::{CaseSeverity, JudgeCaseSpec, RubricSpec, SpecError, SplitPolicySpec, SuiteSpec};
