# EduPilot Membership Inference Attack Documentation

---

## Changes made for the resubmission
* Removed all the extra files from the 'code' folder
* Made the 'How to run the code' section in the README file clearer and more readable
* Added a "What we're doing in this notebook" cell block on top of each .ipynb notebook
* Replaced the EduPilot_690F_git.ipynb and baselines_assignment1.ipynb notebooks with EduPilot_690F_git_updated.ipynb and baselines_assignment1_updated.ipynb respectively.
* Made a lot changes to the inline comments in both the notebooks
* Added the 'Changes made for the resubmission' section on top of the README file
* Added "What we did in the two .ipynb notebooks" section in the README

---

## What we did in the two .ipynb notebooks
* **EduPilot_690F_git_updated.ipynb:**
  1. Recreate the leak-safe text column.
  2. Train a target TF-IDF + 2-layer MLP.
  3. Run a simple Threshold MIA (score = −loss).
  4. Run LiRA with many shadow MLPs (per-example IN/OUT loss modeling).
  5. Compare ROC curves, especially at low FPRs.

* **baselines_assignment1_updated.ipynb:**
  1. Install and import libraries, set seeds, and detect device (CPU/GPU).
  2. Load the EduPilot dataset and build a leak-safe text field called `text_safe` by removing round names and excluding `mock_question`.
  3. Train a TF-IDF + Logistic Regression baseline on `text_safe` and record clean metrics (accuracy, log-loss).
  4. Compute per-example losses (−log p(true)) for Logistic Regression on both train (members) and test (non-members).
  5. Prepare BERT: tokenize `text_safe`, make a small Dataset/Collator, fine-tune a classifier head on top of `bert-base-uncased`, and evaluate it.
  6. Compute per-example losses for BERT (manual batched forward pass).
  7. Run a simple Threshold MIA (score = −loss) for both models and compare ROC curves + TPR at low FPR (0.1, 0.01, 0.001).
  8. Implement LiRA for Logistic Regression: train many shadow models, collect IN vs OUT loss distributions per example, fit Gaussians, and compute LLR scores.
  9) Visualize LiRA internals:
    
      a) pick high/mid/low LLR examples and plot their IN vs OUT histograms,
     
      b) pool all IN vs OUT losses and show hist/boxplots.
     
  10. Compare Threshold MIA vs LiRA (LogReg) on one plot to see how LiRA improves membership detection, especially at low FPR.
  
  So basically, we build leak-safe text, train LR and BERT, get per-example losses, run Threshold MIA on both, then run LiRA on LR, and visualize/compare results.

## Table of Contents

* [Dataset](#dataset)
* [Attack Details](#attack-details)

  * [Implementation](#implementation)
  * [Model-wise Results](#model-wise-results)
  * [Plots](#plots)
  * [Vulnerability Analysis](#vulnerability-analysis)
  * [Implications for EduPilot](#implications-for-edupilot)
* [How to Run the Code](#how-to-run-the-code)
* [LLM usage and References](#llm-usage-and-references)

---

## Dataset

* **Use case:** AI-powered job-seeker interview preparation (EduPilot).

  EduPilot is designed to help job-seekers practice for interviews by generating realistic mock questions across different rounds — Online Assessment (OA), Technical, System Design, HR/Behavioral, and ML Case Study. The system takes in a candidate’s role, company, and location to tailor questions, simulating real interview conditions. The idea is to use large language models (LLMs) to provide personalized interview practice at scale, while handling sensitive user queries like resumes, past experiences, and role-specific skills.
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
  "technical_question": "Implement an LRU cache with O(1) operations."
}
```

---

## Attack Details

### Implementation

**General Attack (MIA & LiRA):**

* After training a model, we record the **per-sample loss** (negative log likelihood of the true class) for both the training set (members) and test set (non-members).
* **Threshold MIA:** If a sample’s loss is much lower, we guess it was in the training set. We then evaluate with ROC/AUC and TPR at strict low FPRs.
* **LiRA:** We train many shadow models(about 256) on random splits. For each example, we collect loss distributions when it is “IN” vs “OUT.” Using these, we compute a **likelihood ratio score** for the target model’s loss. This gives a more reliable signal, especially at low FPRs.

**Model-specific Implementations:**

* **Logistic Regression (TF-IDF):**

  * Text is vectorized with a TF-IDF vectorizer (1–3 grams, 40k features).
  * We train a multinomial Logistic Regression model with the `saga` solver, max\_iter = 4000, and C = 0.5.
  * After training, we directly compute per-sample log-losses for both train and test examples.

* **BERT (bert-base-uncased):**

  * Text is tokenized using HuggingFace’s AutoTokenizer with max length = 192.
  * We fine-tune `bert-base-uncased` for 3 epochs using the HuggingFace Trainer API (batch size 16 train, 32 eval, lr = 2e-5).
  * After training, we evaluate and collect per-sample cross-entropy losses from the logits.

* **MLP (TF-IDF + 2-layer neural net):**

  * Text is vectorized with the same TF-IDF setup (1–3 grams, 40k features).
  * A 2-layer MLP with 512 hidden units is trained in PyTorch for 18 epochs, batch size 64, learning rate 3e-4.
  * No regularization (weight\_decay=0.0) is used, to allow overfitting.
  * After training, we compute per-sample losses for all train and test samples.

---

### Model-wise Results

#### Logistic Regression (TF-IDF)

* **Train mean loss:** 1.41
* **Test mean loss:** 1.63
* **Threshold MIA:** AUC = 0.776

  * TPR\@FPR≤0.1 = 0.41
  * TPR\@FPR≤0.01 = 0.08
  * TPR\@FPR≤0.001 = 0.03
* **LiRA (LogReg):** AUC = 0.8761

  * TPR\@FPR≤0.1 = 0.6679
  * TPR\@FPR≤0.01 = 0.2614
  * TPR\@FPR≤0.001 = 0.0250

*LogReg leaks membership information because it overfits more. But at very low FPR, LiRA shows its real power is weaker than what AUC suggests.*

---

#### BERT (bert-base-uncased)

* **Train mean loss:** 1.598
* **Test mean loss:** 1.608
* **Threshold MIA:** AUC = 0.536

  * TPR\@FPR≤0.1 = 0.15
  * TPR\@FPR≤0.01 = 0.016
  * TPR\@FPR≤0.001 = 0.008

*BERT hardly leaks membership under this attack (AUC ≈ 0.53, close to random). It generalizes well, but we know from other papers that it can still memorize rare sequences, which is a different kind of risk (data extraction).*

---

#### MLP (TF-IDF + 2-layer NN)

* **Train mean loss:** 0.63
* **Test mean loss:** 1.96
* **Threshold MIA:** AUC = 0.934

  * TPR\@FPR≤0.1 = 0.77
  * TPR\@FPR≤0.01 = 0.35
  * TPR\@FPR≤0.001 = 0.079
* **LiRA (MLP):** AUC = 0.804

  * TPR\@FPR≤0.1 = 0.56
  * TPR\@FPR≤0.01 = 0.18
  * TPR\@FPR≤0.001 = 0.02

*The MLP is the leakiest of all — it heavily overfits and threshold MIA looks extremely strong. LiRA again shows that once you restrict to very low FPRs, the attack’s effectiveness is lower than AUC suggests.*

---

### Plots

#### MLP (Simple Threshold MIA and LiRA):

<img width="627" height="562" alt="image" src="https://github.com/user-attachments/assets/9ce89de4-0c6e-4bca-be7a-2649cc546403" />


#### LogReg (Simple Threshold MIA and LiRA):

<img width="684" height="608" alt="image" src="https://github.com/user-attachments/assets/b2ab6493-80b6-491c-8700-f1da226d5644" />


#### LogReg vs BERT (Simple Threshold MIA):

<img width="732" height="605" alt="image" src="https://github.com/user-attachments/assets/cc764207-5a62-4c84-bb87-fd877c001aff" />

---

### Vulnerability Analysis

* **LogReg:** leaks moderately, AUC \~0.77.
* **BERT:** almost no leakage under threshold MIA, but still vulnerable to memorization of rare strings.
* **MLP:** leaks the most (AUC \~0.93) due to strong overfitting.

The models are clearly leaking membership information because of the large overfit gap — the attack could guess many training examples correctly. But the results also match the paper’s warning: AUC by itself is misleading. At stricter FPRs, the attack’s power falls off quickly. This shows why LiRA is a better approach, since it was built to work well in the low-FPR regime.

---

### Implications for EduPilot

* For EduPilot, this means if real job-seeker data (like resumes or candidate questions) were used, with a simple LogReg or MLP, attackers could run membership inference and find out if someone’s data was used. That’s a serious privacy risk.
* BERT looks safer in MIA, but rare or unique data (like a one-off interview question) could be extracted almost verbatim, as Carlini et al. showed for large language models.

--> To protect against this, EduPilot would need defenses such as:

* Regularization to reduce overfitting
* Limiting output exposure (avoid giving out confidence scores)
* Strongest: **Differential Privacy (DP-SGD)**

Without these, the system could leak sensitive candidate information.

---

## How to Run the Code

### MIA and LiRA on MLP:
1. Open the Colab notebook from this repository (`EduPilot_690F_git_updated.ipynb`).
2. Install the required libraries (the notebook already contains `pip install` cells for **transformers**, **datasets**, **torch**, **scikit-learn**, **matplotlib**, **seaborn**, etc.).
3. Run the notebook cells in order. The dataset will be loaded with:

Note: the dataset is already placed under the same folder i.e. "EduPilot_dataset_2000.csv", so no need to make any change.

   ```python
   df = pd.read_csv("EduPilot_dataset_2000.csv")
   ```
4. Follow the notebook to train MLP, run membership inference attacks (Threshold + LiRA), and reproduce the reported metrics.

### MIA and LiRA on LogReg and MIA on BERT:
1. Open the Colab notebook from this repository (`baselines_assignment1_updated.ipynb`).
2. Install the required libraries (the notebook already contains `pip install` cells for **transformers**, **datasets**, **torch**, **scikit-learn**, **matplotlib**, **seaborn**, etc.).
3. Run the notebook cells in order. The dataset will be loaded with:

Note: the dataset is already placed under the same folder i.e. "EduPilot_dataset_2000.csv", so no need to make any change.

   ```python
   df = pd.read_csv("EduPilot_dataset_2000.csv")
   ```
4. Follow the notebook to train LogReg and BERT, run membership inference attacks (Threshold + LiRA), and reproduce the reported metrics.

---

## LLM Usage and References

### How We Used LLM

* Strongly referenced classifier code templates from previous AI-Core courses — NN (682), Advanced NLP (685), ML (589), and NLP (485). The HW1\_example Colab file provided by June gave us a great starting structure for how to organize our notebooks.
* We used a Large Language Model (LLM) to tailor classifier code (Logistic Regression, MLP, BERT) to our dataset and startup scenario.The LLM helped us format training loops, per-example loss collection, and dataset splits.
* When carefully prompted with our requirements, the LLM also generated visualization code (ROC curves, log-loss plots, TPR/FPR metrics). This made it easier for us to debug and interpret the attacks.
* It was also used to get information from the provided research papers using the Q/A approach taught by the instrcutor. LLM was given the answers to the report questions and the summaries to check/verify from the paper one last time before submission. The small tweaks it suggested were sometimes adopted. It was asked to rewrite some answers and summaries in a better and polished way and sometimes asked to add a few more detailes to them in case they felt really short.

### What We Referenced From Papers

* From *Membership Inference Attacks From First Principles* (Carlini et al., 2022):

  * The importance of evaluating MIAs not just by AUC but by True Positive Rate at very low False Positive Rates (≤0.1%).
  * The LiRA method: training shadow models, collecting “IN” vs “OUT” loss distributions for each example, and computing likelihood ratio scores.
    --> Link: https://arxiv.org/abs/2112.03570?

### What We Referenced From Public Code Repositories

* **GitHub repo mia\_attacks:** we looked at its structure for shadow model training and loss recording.
  --> Link: [github.com/superdianuj/mia\_attacks](https://github.com/antibloch/mia_attacks?)
* **GitHub repo lira-pytorch:** we referred to how it computes per-example TPR at low FPR and organizes multiple shadow runs.
  --> Link: [github.com/orientino/lira-pytorch](https://github.com/orientino/lira-pytorch?)

### What We Did Ourselves

* Implemented our own code for Logistic Regression, BERT fine-tuning, and 2-layer MLP models.
* Collected and processed the synthetic dataset (job-seeker queries), implemented safe text preprocessing, and designed evaluation splits.
* Implemented threshold MIA and LiRA scoring directly in our own code.
* Plotted results (ROC curves, TPR/FPR tables), analyzed vulnerabilities.
* Built the presentation + report.
* Added detailed explanations directly into the code cells, describing the design choices, the flow of logic between cells, and how each implementation step connects to the overall project.
