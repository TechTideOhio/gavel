import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

function computeHash(prevHash, record) {
  const hasher = crypto.createHash('sha256');
  hasher.update(Buffer.from('tideos-gavel-judgment-v1\0', 'utf8'));

  const writeLengthPrefixed = (str) => {
    const buf = Buffer.from(str, 'utf8');
    const lenBuf = Buffer.alloc(8);
    lenBuf.writeBigUInt64BE(BigInt(buf.length));
    hasher.update(lenBuf);
    hasher.update(buf);
  };

  writeLengthPrefixed(prevHash);
  writeLengthPrefixed(record.example_id);
  writeLengthPrefixed(record.split);
  writeLengthPrefixed(record.input_prompt);
  writeLengthPrefixed(record.candidate_response);
  writeLengthPrefixed(record.human_label);

  if (record.judge_prediction) {
    writeLengthPrefixed(record.judge_prediction.verdict);
    writeLengthPrefixed(record.judge_prediction.critique);
  }

  return hasher.digest('hex');
}

function makeDataset(prefix, topic, positiveAcc, negativeAcc) {
  const records = [];
  // 120 examples: 60 pass, 60 fail
  // We want exact splits:
  // train: 24 (12 pass, 12 fail) -> ids 1..12 (pass), 13..24 (fail)
  // dev: 48 (24 pass, 24 fail) -> ids 25..48 (pass), 49..72 (fail)
  // test: 48 (24 pass, 24 fail) -> ids 73..96 (pass), 97..120 (fail)

  for (let i = 1; i <= 120; i++) {
    const id = `${prefix}-${String(i).padStart(3, '0')}`;
    let split = 'test';
    let isPass = false;

    if (i <= 12) {
      split = 'train';
      isPass = true;
    } else if (i <= 24) {
      split = 'train';
      isPass = false;
    } else if (i <= 48) {
      split = 'dev';
      isPass = true;
    } else if (i <= 72) {
      split = 'dev';
      isPass = false;
    } else if (i <= 96) {
      split = 'test';
      isPass = true;
    } else {
      split = 'test';
      isPass = false;
    }

    const human_label = isPass ? 'PASS' : 'FAIL';
    const input_prompt = `Evaluate ${topic} submission ${id} against security and quality criteria.`;
    const candidate_response = isPass
      ? `Verified tenant boundary isolation and strict auth checks for ${id}. All assertions pass.`
      : `Bypassed checks or missed tenant isolation in ${id}. Permissive policy applied.`;

    // Judge predictions
    let judgeVerdict = human_label;
    if (isPass) {
      // simulate accuracy
      if (split === 'test') {
        // out of 24, positiveAcc is e.g. 23 (95.8%)
        const testIdx = i - 72; // 1..24
        if (testIdx > positiveAcc) {
          judgeVerdict = 'FAIL';
        }
      }
    } else {
      if (split === 'test') {
        const testIdx = i - 96; // 1..24
        if (testIdx > negativeAcc) {
          judgeVerdict = 'PASS';
        }
      }
    }

    const critique = judgeVerdict === 'PASS'
      ? `Criterion satisfied: response thoroughly verifies ${topic} requirements without omissions for ${id}.`
      : `Rubric violation: response fails tenant boundary isolation and violates ${topic} rules for ${id}.`;

    const raw_response = JSON.stringify({ critique, verdict: judgeVerdict });
    const critique_offset = raw_response.indexOf('"critique"');
    const verdict_offset = raw_response.indexOf('"verdict"');

    const rec = {
      example_id: id,
      split,
      input_prompt,
      candidate_response,
      human_label,
      annotator_id: 'alex-cinovoj',
      human_notes: `Hand-labeled gold standard example for ${topic} evaluation.`,
      judge_prediction: {
        critique,
        verdict: judgeVerdict,
        raw_response,
        critique_offset,
        verdict_offset,
      },
      prev_hash: '',
      record_hash: '',
    };
    records.push(rec);
  }

  // Chain hashes
  let prev = 'genesis';
  for (const r of records) {
    r.prev_hash = prev;
    r.record_hash = computeHash(prev, r);
    prev = r.record_hash;
  }

  return records;
}

const outDir = 'Tools/gavel/packs/fixtures';
fs.mkdirSync(outDir, { recursive: true });

// 1. VSP RLS (23/24 positive = 95.8%, 22/24 negative = 91.7%)
const vspRecs = makeDataset('vsp-rls', 'Row-Level Security', 23, 22);
fs.writeFileSync(path.join(outDir, 'vsp_rls_judge_120.jsonl'), vspRecs.map(r => JSON.stringify(r)).join('\n') + '\n');

// 2. TechTideAI2 Clarification (22/24 positive = 91.7%, 22/24 negative = 91.7%)
const tt2Recs = makeDataset('tt2-clarify', 'clarification constraint', 22, 22);
fs.writeFileSync(path.join(outDir, 'techtideai2_clarification_judge_120.jsonl'), tt2Recs.map(r => JSON.stringify(r)).join('\n') + '\n');

// 3. Harness Postmortem (22/24 positive = 91.7%, 21/24 negative = 87.5%)
const harnessRecs = makeDataset('harness-post', 'handoff postmortem reproduction', 22, 21);
fs.writeFileSync(path.join(outDir, 'harness_postmortem_judge_120.jsonl'), harnessRecs.map(r => JSON.stringify(r)).join('\n') + '\n');

// Counterfactual 1: Verdict before critique
const badOrderRecs = JSON.parse(JSON.stringify(vspRecs));
for (let i = 0; i < 5; i++) {
  const r = badOrderRecs[i];
  const raw = JSON.stringify({ verdict: r.judge_prediction.verdict, critique: r.judge_prediction.critique });
  r.judge_prediction.raw_response = raw;
  r.judge_prediction.verdict_offset = raw.indexOf('"verdict"');
  r.judge_prediction.critique_offset = raw.indexOf('"critique"');
}
let prev1 = 'genesis';
for (const r of badOrderRecs) {
  r.prev_hash = prev1;
  r.record_hash = computeHash(prev1, r);
  prev1 = r.record_hash;
}
fs.writeFileSync(path.join(outDir, 'negative_verdict_before_critique.jsonl'), badOrderRecs.map(r => JSON.stringify(r)).join('\n') + '\n');

// Counterfactual 2: Split leakage (few-shot contains test ID)
// We write a fixture that has a test item labeled as train or few-shot leaked
const leakedRecs = JSON.parse(JSON.stringify(vspRecs));
// leak a test item into train split with same id
leakedRecs[75].example_id = 'vsp-rls-001'; // duplicate ID across splits
let prev2 = 'genesis';
for (const r of leakedRecs) {
  r.prev_hash = prev2;
  r.record_hash = computeHash(prev2, r);
  prev2 = r.record_hash;
}
fs.writeFileSync(path.join(outDir, 'negative_split_leakage.jsonl'), leakedRecs.map(r => JSON.stringify(r)).join('\n') + '\n');

// Counterfactual 3: Low TNR sycophant (passes 23/24 positives, but only 12/24 negatives, TNR = 50%)
const sycophantRecs = makeDataset('vsp-rls-syco', 'Row-Level Security', 23, 12);
fs.writeFileSync(path.join(outDir, 'negative_low_tnr_sycophant.jsonl'), sycophantRecs.map(r => JSON.stringify(r)).join('\n') + '\n');

// Counterfactual 4: Under 100 examples (only 40 examples)
const shortRecs = vspRecs.slice(0, 40);
let prev4 = 'genesis';
for (const r of shortRecs) {
  r.prev_hash = prev4;
  r.record_hash = computeHash(prev4, r);
  prev4 = r.record_hash;
}
fs.writeFileSync(path.join(outDir, 'negative_under_100_examples.jsonl'), shortRecs.map(r => JSON.stringify(r)).join('\n') + '\n');

console.log('Fixtures generated successfully.');
