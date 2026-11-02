# Contributing to gavel

Thank you for contributing to `gavel`, Tool 9 of the TideOS Eval Kit!

## Development Guidelines

1. **Pure Rust Standard**: Zero C-FFI, `#![forbid(unsafe_code)]`, and zero `.unwrap()` in library and runtime paths.
2. **Deterministic Gating**: All evaluations, scoring functions, and split generators must be 100% deterministic given identical inputs and PRNG seeds.
3. **Typography**: Soft Factory enforces zero em-dashes and zero en-dashes across all code and documentation.
4. **Gates**:
   - `cargo fmt --all -- --check`
   - `cargo clippy --all-targets -- -D warnings`
   - `cargo test --all-targets`
   - `gavel repo .` must pass all 8 checks.
