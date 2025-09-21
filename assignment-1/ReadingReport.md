# Paper Review Documentation

---

## Table of Contents

* [Paper: Membership Inference Attacks from First Principles (LiRA)](#paper-membership-inference-attacks-from-first-principles-lira)

  * [Summary](#summary)
  * [Strength](#strength)
  * [Weakness](#weakness)
* [Connect](#connect)

  * [How do MIAs exploit overfitting?](#how-do-mias-exploit-overfitting)
  * [How does data extraction differ from MIAs, and why are LLMs vulnerable?](#how-does-data-extraction-differ-from-mias-and-why-are-llms-vulnerable)
  * [Is your dataset vulnerable to MIA? Why?](#is-your-dataset-vulnerable-to-mia-why)
  * [What privacy protections does FL claim, and what risks remain?](#what-privacy-protections-does-fl-claim-and-what-risks-remain)
  * [What new attack surface does FL introduce?](#what-new-attack-surface-does-fl-introduce)
  * [Would FL improve privacy for your project? Why/why-not?](#would-fl-improve-privacy-for-your-project-whywhynot)
* [Discussion](#discussion)

  * [What properties of training data make extraction more dangerous?](#what-properties-of-training-data-make-extraction-more-dangerous)
  * [Is FL primarily a privacy technique or a scalabilityavailability technique?](#is-fl-primarily-a-privacy-technique-or-a-scalabilityavailability-technique)

---

## Paper: Membership Inference Attacks from First Principles (LiRA)

### Summary

This paper talks about membership inference attacks (MIAs). These are attacks where someone tries to figure out if a piece of data was used to train a model. The authors say most older methods measured success in the wrong way (just by accuracy or AUC) and need to be evaluated at very low false positive rates (FPRs). They introduce **LiRA (Likelihood Ratio Attack)** which models per-sample difficulty using Gaussian distributions of loss from shadow models. It works much better, showing up to **10× better true positive rates (TPRs)** than prior MIAs. So, if we care about having almost no false alarms, this is it.

### Strength

The paper makes the problem more scientific by treating it as a hypothesis test. It clarifies the shortcomings of evaluating MIAs with average AUC by explaining why low false positives are important.

### Weakness

Running LiRA can be very expensive because you need to train a lot of shadow models to approximate distributions, which is not easy for big models.

---

## Connect

### How do MIAs exploit overfitting?

If a model memorizes training data too much, it gives lower loss or higher confidence for those training examples than for new ones and this gap is detectable. We implemented an MIA by first training a slightly overfit TF-IDF + MLP model on job-seeker queries, and then distinguishing train vs. test samples using a loss-based threshold that directly leveraged the overfitting gap.

We trained a TF-IDF + MLP model on job-seeker queries that slightly overfit, then used the difference in loss values to tell apart training samples from test samples.

### How does data extraction differ from MIAs, and why are LLMs vulnerable?

MIA only tells if some data was used in training. Data extraction actually tries to pull the data itself out of the model. LLMs are weak here because they sometimes memorize rare/unique things like emails, IDs, or unique interview questions which can later be regurgitated with carefully crafted prompts.

### Is your dataset vulnerable to MIA? Why?

Yes — our synthetic EduPilot dataset intentionally allowed mild overfitting (train acc \~87%, test acc \~20%). This created a large generalization gap that MIAs could exploit and easily spot training examples, achieving ROC-AUC ≈ 0.93.

### What privacy protections does FL claim, and what risks remain?

FL says it protects privacy because the raw data (like resumes or queries) never leaves the user’s device. Instead, only model updates are shared. But the paper shows that this doesn’t fully solve the problem – attackers can sometimes reverse-engineer updates (through gradient inversion) or still do membership inference attacks on the model.

### What new attack surface does FL introduce?

Since training happens across many devices, FL is open to poisoning or backdoor attacks. For example, malicious clients can upload bad updates and influence the global model. The paper also talks about attacks where updates leak information about the original data.

### Would FL improve privacy for your project? Why/Why-not?

It might help a little because job-seeker data wouldn’t all be stored on one server. But it’s not enough on its own, since updates could still leak and poisoned updates could hurt the model. So, for EduPilot, FL by itself won’t solve privacy problems – we’d still need something stronger like differential privacy or secure aggregation.

---

## Discussion

### What properties of training data make extraction more dangerous?

If the training data is rare, unique, or very sensitive (like an SSN, email address, or a one-of-a-kind interview question), then the model is more likely to memorize it. This makes extraction much more dangerous because the exact data can come out of the model. Common or repeated data are less risky since they are harder to link back to one person.

### Is FL primarily a privacy technique or a scalability/availability technique?

From what the paper says, FL was mainly created as a way to scale training across many devices and avoid sending all the data to a central server. It does give some privacy benefits because the raw data stays on the device, but by itself it’s not a full privacy solution. So I’d say it’s more of a scalability/availability technique, with privacy being a partial side benefit.

---

