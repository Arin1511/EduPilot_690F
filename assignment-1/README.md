# EduPilot Membership Inference Attack Documentation

---

## Table of Contents

* [Dataset](#dataset)
* [Attack Details](#attack-details)

  * [Design Choices](#design-choices)
  * [Implementation](#implementation)
  * [Metrics & Results](#metrics--results)
  * [Vulnerability Analysis](#vulnerability-analysis)
  * [Implications](#implications)

---

## Dataset

* **Use case:** Job-seeker interview preparation (EduPilot).
* **#samples:** 2000 total examples.
* **Label distribution:** 5 interview rounds (OA, Technical, System Design, HR/Behavioral, ML Case Study) — balanced across categories.
* **Generation method:** Synthetic dataset with job queries, roles, companies, and generated mock interview questions. The questions were referenced from neetcode. Furthermore, a “safe text” field was created by stripping round-indicative keywords to prevent trivial leakage.

### Example

```json
{
  "user_query": "Give me mock questions for Software Engineer role at Google NYC",
  "job_role": "Software Engineer",
  "company": "Google",
  "location": "NYC",
  "interview_round": "Technical",
  "mock_question": "Implement an LRU cache with O(1) operations."
}
```

---

## Attack Details

### Design Choices

* **Feature extraction:** TF-IDF vectorizer with 1–3 grams and max 40,000 features.
* **Model:** 2-layer MLP (512 hidden units) — intentionally a bit large to encourage mild overfitting.
* **Training setup:** 18 epochs, batch size 64, learning rate 3e-4, weight decay 0.0.
* **Data split:** 70% train, 30% test, stratified by interview round.
* **Overfitting on purpose:** No strong regularization so the model leaks more membership information.

### Implementation

* **Model choice:** Used TF-IDF features with a 2-layer MLP classifier.
* **Training setup:** Allowed the model to overfit on purpose by not adding strong regularization.
* **Data collection:** After training, recorded the loss for every sample in both the training set and the test set.
* **Attack method:** Applied a simple Shokri-style threshold MIA — if a sample had lower loss (looked “easier” for the model), it was guessed to be in the training set.
* **Evaluation metrics:** Plotted the ROC curve, calculated the AUC, and measured true positive rate (TPR) at very low false positive rates (FPRs).

### Metrics & Results

* Train accuracy: \~87%
* Test accuracy: \~20% → big generalization gap (overfit)
* ROC-AUC: 0.93 (looks strong at first glance)
* TPR @ FPR ≤ 0.1: 0.77
* TPR @ FPR ≤ 0.01: 0.31
* TPR @ FPR ≤ 0.001: 0.06

 This means the attack seems very good if you only look at AUC, but once you care about keeping false positives very low, the success rate drops a lot.

---

## Vulnerability Analysis

The model is clearly leaking membership information because of the large overfit gap — the attack could guess many training examples correctly. But the results also match the paper’s warning: AUC by itself is misleading. At stricter FPRs, the attack’s power falls off quickly. This shows why **LiRA** is a better approach, since it was built to work well in the low-FPR regime.

---

## Implications

For EduPilot, this means if real job-seeker data (like resumes or candidate questions) were used, attackers could run membership inference and find out if someone’s data was used. That’s a serious privacy risk. Even worse, rare or unique data (like a one-off interview question) could be extracted almost verbatim, as Carlini et al. showed for large language models.

➡ To protect against this, EduPilot would need defenses such as:

* Regularization
* Limiting outputs
* Stronger methods like differential privacy

Without these, the system could leak sensitive candidate information.

---


Purpose: this repo/notebook collection implements two membership-inference analyses on the EduPilot dataset:

LiRA on Logistic Regression (LR) — Likelihood Ratio Attack using shadow LR models.
Result: LiRA AUC ≈ 0.741 (≈ 74.1%).

MIA on MLP — several MIA flavors tested against an MLP victim:
Summary AUCs (MLP):

Threshold: 0.9305

AttackClf: 0.5326

LabelOnly: 0.5177

This README explains what these numbers mean, how to reproduce the runs, how to interpret results, troubleshooting, and recommended next steps / mitigations.

