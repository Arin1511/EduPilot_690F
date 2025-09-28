# EduPilot Federated Learning Documentation

---

## Table of Contents

* [Dataset](#dataset)
* [Design Choices](#design-choices)
* [Model & Training Details](#model--training-details)
* [Client Partitioning](#client-partitioning)
* [Aggregation Methods](#aggregation-methods)
* [Performance Comparison](#performance-comparison)
* [Vulnerabilities & Implications](#vulnerabilities--implications)
* [How to Run the Code](#how-to-run-the-code)
* [References](#references)

---

## Dataset

* **Use case:** AI-powered job-seeker interview preparation (EduPilot).

EduPilot is designed to help job-seekers practice for interviews by generating realistic mock questions across different rounds — Online Assessment (OA), Technical, System Design, HR/Behavioral, and ML Case Study. The system takes in a candidate’s role, company, and location to tailor questions, simulating real interview conditions. The idea is to use large language models (LLMs) to provide personalized interview practice at scale, while handling sensitive user queries like resumes, past experiences, and role-specific skills.
* **#samples:** 2000 total synthetic datasets.
* **Label distribution:** 5 interview rounds — balanced across categories.

  * Online Assessment (OA)
  * Technical
  * System Design
  * HR / Behavioral
  * ML Case Study
* **Generation method:** Synthetic dataset with job queries, roles, companies, and generated mock interview questions. The questions were referenced from neetcode. Furthermore, a “safe text” field was created by stripping round-indicative keywords to prevent trivial leakage.
**Example entry:**

```json
{
  "user_query": "Give me mock questions for Software Engineer role at Google NYC",
  "job_role": "Software Engineer",
  "company": "Google",
  "location": "NYC",
  "interview_round": "Technical",
  "technical question": "Implement an LRU cache with O(1) operations."
}
```

---

## Design Choices

* **#Clients:** 5
  We used 5 clients — enough to feel realistic, but still easy to run on our machines.
*  **Model:** Neural Network
* **IID simulation:** Stratified 5-fold split ensures that each client has the same mix of labels as the full dataset.
* **Non-IID simulation:** Label-skew strategy; we gave each client mostly 2 types of labels(this makes them biased towards ~2 labels), with the rest spread out randomly.
* **Local epochs:** Set to **5** — long enough to let local models learn, short enough to avoid divergence.
* **Rounds:** 100 federated rounds to observe convergence patterns.
* **Hidden units:** 64 for FL (to reduce comms cost); 128 for centralized baseline.
* **Regularization:** L2 with λ = 1e-4, excluding bias terms.

---

## Model & Training Details

* **Architecture:** Custom NumPy MLP

  * Input: TF-IDF vectors (1–2 grams, max 2000 features)
  * Hidden: 1 fully-connected layer with sigmoid activations
  * Output: Softmax over 5 classes
* **Initialization:** He normal, bias included
* **Optimizer:** Plain SGD
* **Learning rate:**

  * Centralized: 0.10
  * Federated (local): 0.05
* **Batch size:** Full batch (all local data per client per epoch)
* **Epochs:**

  * Centralized: 100 outer loops (each inner loop trains for 3 epochs)
  * Federated: 5 epochs per client per round

---

## Client Partitioning

| Mode        | Partition Strategy                        | Notes                                          |
| ----------- | ----------------------------------------- | ---------------------------------------------- |
| **IID**     | StratifiedKFold (5 splits)                | Balanced label distribution across clients     |
| **Non-IID** | Label skew (≈2 labels/client + leftovers) | Creates realistic heterogeneity / client drift |

**We have added histograms for client 1 to show the distribution:**

![img_4.png](img_4.png)

* *IID*: Here, for client-1 the distributions among 5 classes are normal
* *non-IID*: But in non-iid, it is biased towards one of classes, shows imbalance and skewed result.

---

## Aggregation Methods

* **FedAvg**
  Weighted average of client weight vectors.
  
  Formula : w^(t+1) = Σ_i [ (n_i / Σ_j n_j) * w_i^(t) ]
  
  Meaning:
   - w^(t+1): the updated value at the next step
   - w_i^(t): the value of component i at the current step
   - n_i: a weight or count associated with i
   - Σ_j n_j: the sum of all n_j values (normalization term)
   - The formula computes a weighted average of w_i^(t), with weights proportional to n_i

* **FedMedian**
  Coordinate-wise median of client weight vectors. Robust to outliers or poisoned updates.

  Formula: w^(t+1) = median( w_1^(t), w_2^(t), ..., w_k^(t) )

  Meaning:
   - Each w_i^(t) is a weight vector from client i at time t
   - For each coordinate (dimension) of the weight vector, take the median value across all clients
   - This produces a new vector w^(t+1), where each element is the median of that coordinate
   - The method is robust to outliers or poisoned updates because extreme values do not affect the median as much as they affect the mean

* **Client simulation details:**

  * Participation: all 5 clients each round (no dropouts)
  * Local training: fresh model copy initialized with current global weights each round

---

## Performance Comparison

### Centralized vs Federated

| Setting             | Final Accuracy | Notes                                               |
| ------------------- | -------------- | --------------------------------------------------- |
| Centralized (128h)  | ~0.85          | Best reference performance                          |
| FedAvg (IID)        | ~0.82–0.83     | Converges close to centralized                      |
| FedMedian (IID)     | ~0.80–0.81     | Slightly slower convergence, more stable            |
| FedAvg (Non-IID)    | ~0.74–0.75     | Accuracy drop due to label skew                     |
| FedMedian (Non-IID) | ~0.73–0.74     | More robust than FedAvg under skew, but still lower |

**Key observations:**

* IID FL nearly matches centralized training.
* Non-IID heterogeneity significantly slows learning and lowers final accuracy.
* FedMedian offers robustness to skew but does not fully recover centralized accuracy.
* Communication overhead: Each round requires clients to send their full parameter vector (~O(#weights)).

---

## Vulnerabilities & Implications

* **Privacy advantage:** Raw candidate text never leaves clients — only weight updates are shared. This is a major step forward compared to centralized training, because sensitive information such as resumes, personal job queries, or role-specific interview experiences never leave the user’s local device. By restricting communication to model updates, EduPilot avoids exposing plain text data to a central server. In a realistic deployment, this would reduce the risk of data breaches, insider threats, or accidental leaks from a centralized database. In short, FL ensures that user data stays with the user.
* **But privacy is not guaranteed:**

  * Gradient/weight updates can still leak information.
  * Membership Inference and data extraction attacks remain possible without extra safeguards.
* **Implications for EduPilot:**

  * FL reduces risk compared to centralized training, but **defenses such as secure aggregation and DP-SGD are needed** for strong protection.
  * Especially important given EduPilot’s sensitive inputs (resumes, job queries, personal experiences).

---

## How to Run the Code

Please follow below instructions to get the comparisons using FedAvg Aggregator.

1. **Step 1: Please execute 'centralize_global_file.py" to generate Centralized baseline & artifacts**

   ```bash
   python assignment-2/centralize_global_file.py
   ```
   Produces:

   * `artifacts_centralized/{tfidf_vectorizer.pkl, label_encoder.pkl, centralized_*_text_labels.csv}`
   * `central_accuracy.csv`

2. **Step 2: Please execute 'federated_learning_iid.py" to generate Federated Learning (FedAvg, IID) result.**

   ```bash
   python assignment-2/federated_learning_iid.py
   # -> fl_iid_accuracy.csv
   ```

3. **Step 3: Please execute 'federated_learning_non_iid.py to generate Federated Learning (FedAvg, Non-IID) result.**

   ```bash
   python assignment-2/federated_learning_non_iid.py
   # -> fl_non_iid_accuracy.csv
   ```

4. **Step 5:Please execute 'graph_plotting.py to generate graph to see the comparison between IID and non-IID**

   ```bash
   python assignment-2/graph_plotting.py
   # -> fl_iid_vs_non_iid_vs_central.png
   ```
Please follow below instructions to get the comparisons using FedMedian Aggregator.

1. **Step 1:Please execute 'fedmedian_iid.py to generate Federated Learning (FedMedian, IID) result.**

   ```bash
   python assignment-2/fedmedian_iid.py
   # -> fl_iid_fedmedian_accuracy.csv
   ```
2. **Step 5:Please execute 'fedmedian_non_iid.py to generate Federated Learning (FedMedian, Non-IID) result.**

   ```bash
   python assignment-2/fedmedian_non_iid.py
   # -> fl_non_iid_fedmedian_accuracy.csv
   ```

3. **Step 5:Please execute 'graph_plotting_fedmedian.py to generate graph to see the comparison between IID and non-IID**

   ```bash
   python assignment-2/graph_plotting_fedmedian.py
   # -> fedMedianPlot.png
   ```
   
Please follow below instructions to get the comparisons using FedSgd Aggregator.

1. **Step 1:Please execute 'fedmedian_iid.py to generate Federated Learning (FedMedian, IID) result.**

   ```bash
   python assignment-2/fedsgd_iid.py
   # -> fedsgd_iid.csv
   ```
2. **Step 5:Please execute 'fedsgd_non_iid.py to generate Federated Learning (FedMedian, Non-IID) result.**

   ```bash
   python assignment-2/fedsgd_non_iid.py
   # -> fedsgd_non_iid.csv
   ```

3. **Step 5:Please execute 'graph_plotting_fedsgd.py to generate graph to see the comparison between IID and non-IID**

   ```bash
   python assignment-2/graph_plotting_fedsgd.py
   # -> fl_iid_vs_non_iid_vs_central_fedsgd.png
   ```

---

## References

* McMahan et al., 2017. *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg).
* Yin et al., 2018. *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates* (FedMedian).
* Carlini et al., 2022. *Membership Inference Attacks from First Principles* (for vulnerabilities).
