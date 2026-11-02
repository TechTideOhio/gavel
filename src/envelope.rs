use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnvelopeStatus {
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessEnvelope<T> {
    pub status: EnvelopeStatus,
    pub summary: String,
    pub next_actions: Vec<String>,
    pub artifacts: Vec<String>,
    pub data: T,
}

impl<T: Serialize> HarnessEnvelope<T> {
    pub fn success(summary: impl Into<String>, data: T) -> Self {
        Self {
            status: EnvelopeStatus::Success,
            summary: summary.into(),
            next_actions: Vec::new(),
            artifacts: Vec::new(),
            data,
        }
    }

    pub fn warning(summary: impl Into<String>, data: T) -> Self {
        Self {
            status: EnvelopeStatus::Warning,
            summary: summary.into(),
            next_actions: Vec::new(),
            artifacts: Vec::new(),
            data,
        }
    }

    pub fn error(summary: impl Into<String>, data: T) -> Self {
        Self {
            status: EnvelopeStatus::Error,
            summary: summary.into(),
            next_actions: Vec::new(),
            artifacts: Vec::new(),
            data,
        }
    }

    #[must_use]
    pub fn with_next_action(mut self, action: impl Into<String>) -> Self {
        self.next_actions.push(action.into());
        self
    }

    #[must_use]
    pub fn with_artifact(mut self, artifact: impl Into<String>) -> Self {
        self.artifacts.push(artifact.into());
        self
    }

    pub fn print_json(&self) -> Result<(), anyhow::Error> {
        let serialized = serde_json::to_string_pretty(self)?;
        println!("{serialized}");
        Ok(())
    }
}
