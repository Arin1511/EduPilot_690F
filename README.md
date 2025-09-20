README — EduPilot Membership Inference Experiments

Purpose: this repo/notebook collection implements two membership-inference analyses on the EduPilot dataset:

LiRA on Logistic Regression (LR) — Likelihood Ratio Attack using shadow LR models.
Result: LiRA AUC ≈ 0.741 (≈ 74.1%).

MIA on MLP — several MIA flavors tested against an MLP victim:
Summary AUCs (MLP):

Threshold: 0.9305

AttackClf: 0.5326

LabelOnly: 0.5177

This README explains what these numbers mean, how to reproduce the runs, how to interpret results, troubleshooting, and recommended next steps / mitigations.