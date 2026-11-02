use crate::classifier::ConfusionMatrix;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeTokenEconomics {
    pub prompt_tokens_per_eval: usize,
    pub critique_tokens_per_eval: usize,
    pub cost_per_million_input: f64,
    pub cost_per_million_output: f64,
}

impl Default for JudgeTokenEconomics {
    fn default() -> Self {
        Self {
            prompt_tokens_per_eval: 1200,
            critique_tokens_per_eval: 85,
            cost_per_million_input: 3.00,
            cost_per_million_output: 15.00,
        }
    }
}

impl JudgeTokenEconomics {
    pub fn compute_eval_cost_usd(&self, total_evals: usize) -> f64 {
        let total_in = (total_evals * self.prompt_tokens_per_eval) as f64;
        let total_out = (total_evals * self.critique_tokens_per_eval) as f64;

        (total_in / 1_000_000.0) * self.cost_per_million_input
            + (total_out / 1_000_000.0) * self.cost_per_million_output
    }

    pub fn compute_risk_adjusted_loss(
        &self,
        matrix: &ConfusionMatrix,
        cost_per_false_positive_usd: f64,
        cost_per_false_negative_usd: f64,
    ) -> f64 {
        let api_cost = self.compute_eval_cost_usd(matrix.total());
        let fp_loss = (matrix.fp as f64) * cost_per_false_positive_usd;
        let fn_loss = (matrix.fn_count as f64) * cost_per_false_negative_usd;

        api_cost + fp_loss + fn_loss
    }
}
