# Changelog

All notable changes to `gavel` will be documented in this file.

## [0.1.0] - 2026-11-01

### Added
- Initial release of `gavel` (TideOS Eval Kit Tool 9).
- Binary classifier alignment engine for LLM-as-a-Judge.
- Stratified train/dev/test dataset partitioning with deterministic SplitMix64 PRNG.
- Zero-leakage split isolation verification and few-shot containment auditing.
- Structural critique-before-verdict byte order and rubric token verification.
- Pinned judge model version and temperature=0.0 determinism enforcement.
- Confusion matrix and alignment metrics: TPR, TNR, FPR, FNR, Youden J, Cohen kappa, and MCC.
- Rogan-Gladen (1978) prevalence bias correction with 95% bootstrap confidence intervals.
- Sequential Merkle judgment hash chaining and sha256-framed-v1 lockfile support.
- 8-point Soft Factory repository hygiene auditor.
- OpenTelemetry, Mermaid, and reval export formats.
