# Plan

Phases are ordered by how much they raise the project's credibility. Do not skip ahead: Phase 1 decides what the rest of the report can claim.

## Phase 0: Audit (read-only)
Understand the code and answer: is the checkpoint BC-only? what does the Teacher see? what is the QoE formula? how are train/test traces split? how is the observation built?
Deliverable: `AUDIT.md`. Acceptance: every question answered with file and line references.

## Phase 1: Ablation (highest priority)
Train and evaluate BC-only, BC+PPO, and PPO from scratch with 5 or more seeds each. Log learning curves (QoE vs training steps, mean and std across seeds).
Deliverable: `results/ablation/` with CSV, curves, and a summary table. Acceptance: one command reproduces it; the summary states plainly whether PPO fine-tuning beats BC-only.

## Phase 2: Diagnose the train/test gap and add statistics
Evaluate every policy on both sets. Print trace counts, throughput mean/median/variance for each set. Report QoE with 95 percent bootstrap CIs, a paired comparison versus MPC and Teacher, per-trace win rate, rebuffer p50/p90, and a QoE component breakdown (bitrate, rebuffer, smoothness). Re-run under two other QoE weightings.
Deliverable: `results/eval/`. Acceptance: the gap has an evidence-backed explanation.

## Phase 3: Datasets and generalization
Add a standard dataset (FCC broadband and/or HSDPA Norway 3G). Include a train-on-one, test-on-other experiment. If there is no network access, document what to download and where.
Deliverable: `results/generalization/`.

## Phase 4: Real live demo
Build the segmented-video demo described in `demo/DEMO_GUIDE.md`: two players, same video, same trace, one driven by the baseline and one by the real PPO checkpoint, with live metrics that reproduce the evaluation numbers.
Deliverable: a one-command launcher plus a recorded fallback video. Acceptance: demo metrics on the stress trace match evaluation within rounding.

## Phase 5: Report and verification
Write the report from files in `results/` only. Fresh-clone reproduction check. Honest limitations section.

## Optional extras (only if time remains)
Second RL baseline (DQN or A3C Pensieve-style), reward-shaping study, state-feature ablation.

## Risks
- BC-only equals BC+PPO: report it; consider tuning PPO (lower learning rate, entropy bonus, longer training) honestly on validation traces.
- Observation mismatch between Python and browser: avoid by letting the demo call a Python policy server (Option A in the guide).
- Live demo instability: rehearse, pre-generate the renditions, keep a recording.
