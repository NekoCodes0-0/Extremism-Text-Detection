# Extremism Detection in Social Media Text
## Top-10 Solution in a Kaggle Competition

---

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![Model](https://img.shields.io/badge/Model-DeBERTa--v3--base-purple.svg)
![Task](https://img.shields.io/badge/Task-Text%20Classification-green.svg)
![Competition](https://img.shields.io/badge/Kaggle-Top%2010-blue.svg)
![License](https://img.shields.io/badge/License-MIT%20Use-lightgrey.svg)
![Dataset](https://img.shields.io/badge/Dataset-Social%20Media%20Extremism-orange.svg)
![Evaluation](https://img.shields.io/badge/Evaluation-Kaggle%20Leaderboard-blue.svg)
![Seeded](https://img.shields.io/badge/Seed-Fixed-success.svg)
![Deterministic](https://img.shields.io/badge/CUDA-Deterministic-blue.svg)
![Inference](https://img.shields.io/badge/Inference-Standalone-green.svg)

---

---

## Abstract

This repository presents a complete deep learning pipeline for detecting extremist content in social media text.  
The task is formulated as a binary classification problem distinguishing between extremist and non-extremist messages.

The solution was developed for a Kaggle competition on digital extremism detection and achieved a **Top-10 position on the public leaderboard**.  
The approach focuses on contextual understanding, robustness to noisy language, and stable generalization rather than keyword-based heuristics.

---

## Table of Contents

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Definition](#2-problem-definition)
3. [Challenges in Extremism Detection](#3-challenges-in-extremism-detection)
4. [Methodology](#4-methodology)
5. [Model Architecture](#5-model-architecture)
6. [Text Preprocessing](#6-text-preprocessing)
7. [Training Strategy](#7-training-strategy)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Experimental Results](#9-experimental-results)
10. [Repository Structure](#10-repository-structure)
11. [Setup and Usage](#11-setup-and-usage)
12. [Limitations](#12-limitations)
13. [Conclusion](#13-conclusion)
14. [Related Work: State-of-the-Art Security ML Systems on GitHub](#14-related-work-state-of-the-art-security-ml-systems-on-github)

---

## 1. Introduction

Social media platforms generate large volumes of user-created content, some of which may contain extremist ideology, propaganda, or incitement.  
Automatic detection of such content is critical but difficult due to language ambiguity, context dependence, and adversarial writing styles.

This project explores a transformer-based solution designed to address these challenges using modern regularization and training techniques.

---

## 2. Problem Definition

Given a short text message from a social media platform, the goal is to classify it into one of two categories:

- **EXTREMIST**
- **NON_EXTREMIST**

The model must rely on contextual meaning rather than isolated keywords.

---

## 3. Challenges in Extremism Detection

### 3.1 Context Dependence

Extremist terms frequently appear in:
- News reporting
- Academic analysis
- Condemnatory statements

Correct classification requires understanding **intent**, not vocabulary alone.

### 3.2 Ambiguous Word Usage

Many words have multiple meanings depending on:
- Cultural context
- Political discourse
- Temporal trends

### 3.3 Noisy and Adversarial Language

Social media text often includes:
- Repeated characters
- Misspellings
- Slang and abbreviations
- Excessive punctuation

These patterns are sometimes intentionally used to bypass automated moderation.

### 3.4 Label Uncertainty

Some samples are inherently ambiguous, leading to:
- Annotation noise
- Model overconfidence
- Reduced generalization

---

## 4. Methodology

The solution is built around a large pretrained language model combined with several stability-focused enhancements:

- Contextual transformer representations
- Layer-wise feature aggregation
- Implicit ensembling via dropout
- Adversarial regularization
- Confidence smoothing during optimization

The goal is not only high accuracy but **robust decision boundaries**.

### Technologies and Tools Used

### Core Frameworks
- **Python** (3.9+)
- **PyTorch** (2.x)
- **Hugging Face Transformers**

### Model Architecture
- **DeBERTa-v3-base**
- Weighted Layer Pooling
- Multi-Sample Dropout
- Adversarial Weight Perturbation (AWP)

### Training Techniques
- AdamW Optimizer
- Cosine Learning Rate Scheduler
- Gradient Accumulation
- Label Smoothing

### Data Handling
- Pandas
- NumPy
- Custom text normalization pipeline

### Hardware
- NVIDIA GPU (single-GPU training)
- Optimized for low VRAM via accumulation

---

## 5. Model Architecture

### 5.1 Backbone

- Model: `microsoft/deberta-v3-base`
- Hidden size: 1024
- Full hidden-state outputs enabled

### 5.2 Layer Pooling

The final representation is obtained by:
- Extracting the `[CLS]` token from the last four transformer layers
- Averaging them to reduce layer-specific noise

### 5.3 Multi-Sample Dropout

- Five dropout layers with increasing dropout probabilities
- The classifier is applied multiple times
- Logits are averaged to form the final prediction

This improves robustness and reduces variance.

### 5.4 Adversarial Weight Perturbation (AWP)

- Small gradient-aligned perturbations applied to model weights
- Activated after initial convergence
- Encourages flatter loss landscapes and improved generalization

---

## 6. Text Preprocessing

A controlled cleaning strategy is applied:

- Lowercasing
- Removal of HTML-like tags
- Normalization of repeated characters
- Removal of excessive punctuation while preserving `?` and `!`
- Whitespace normalization

The intent is to reduce noise without destroying semantic signals.

---

## 7. Training Strategy

### 7.1 Optimization

- Optimizer: AdamW
- Learning rate: 1e-5
- Weight decay: 0.01

### 7.2 Scheduling

- Cosine learning rate schedule
- Warmup over the first 10% of training steps

### 7.3 Regularization

- Label smoothing (0.05)
- Multi-sample dropout
- Adversarial weight perturbation

### 7.4 Memory Efficiency

- Small batch size with gradient accumulation
- Enables training on GPUs with limited VRAM

---

## 8. Inference Pipeline

Inference is separated from training and submission logic.

### Input
A CSV file containing:
- `Original_Message`

### Output
A CSV file with an additional column:
- `Prediction` (`EXTREMIST` or `NON_EXTREMIST`)

The inference script performs preprocessing, tokenization, and classification in evaluation mode.

---

## 9. Experimental Results

- Achieved **Top-10 placement** in a Kaggle competition
- Stable validation performance across folds
- Improved robustness to noisy and adversarial samples
- Reduced overconfidence on borderline cases

---

## 10. Repository Structure

```text
.
├── train.py          # Training pipeline
├── inference.py      # Standalone inference script
├── model.py          # Model definition
├── requirements.txt  # Dependencies
└── README.md
```

## 11. Setup & Usage

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```
## 12. Limitations
`
👉The model may struggle with extremely short or context-free messages
👉Cultural and regional slang not present in training data can affect accuracy
👉Binary classification does not capture degrees or categories of extremism
`
## 13. Conclusion

This project demonstrates that robust extremism detection requires more than keyword filtering.
Contextual modeling, regularization, and disciplined training strategies are essential for handling noisy, real-world social media text.

The results validate the effectiveness of combining large pretrained transformers with stability-focused enhancements.

---

## 14. Related Work: State-of-the-Art Security ML Systems on GitHub

The techniques in this repository (transformer-based classification, adversarial regularization, and robust text preprocessing) share significant methodological overlap with modern **banking and financial security systems**.  
Both domains frame the core problem identically: given noisy, adversarial input, reliably classify or flag harmful patterns.

The repositories below represent the current state of the art in security-focused machine learning, including banking fraud detection, anti-money-laundering (AML), and adversarial robustness.

---

### 14.1 Anti-Money Laundering & Financial Fraud Detection

| Repository | Description | Key Techniques |
|---|---|---|
| [IBM/AMLSim](https://github.com/IBM/AMLSim) | Scalable anti-money-laundering transaction simulation and detection framework by IBM Research | Graph-based modelling, rule injection, GNN classifiers |
| [Feedzai/bank-account-fraud](https://github.com/feedzai/bank-account-fraud) | Benchmark dataset and baseline models for bank-account fraud detection (NeurIPS 2022 Dataset Track) | Tabular ML, fairness evaluation, imbalanced classification |
| [dkaslovsky/Coupled-Gaussian-Mixture-Model](https://github.com/dkaslovsky/Coupled-Gaussian-Mixture-Model) | Unsupervised anomaly detection for financial transactions using coupled Gaussian mixture models | GMMs, EM algorithm, statistical threshold calibration |

---

### 14.2 Adversarial Robustness (directly applicable to this project)

| Repository | Description | Key Techniques |
|---|---|---|
| [Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) | IBM's comprehensive adversarial ML defence library, widely used in banking security pipelines | AWP, PGD, Carlini-Wagner attacks & defences, certified robustness |
| [cleverhans-lab/cleverhans](https://github.com/cleverhans-lab/cleverhans) | Benchmark adversarial-examples library by Ian Goodfellow and Nicolas Papernot | FGSM, Madry PGD, Virtual Adversarial Training |
| [QData/TextAttack](https://github.com/QData/TextAttack) | Framework for adversarial attacks, data augmentation, and adversarial training in NLP | BERT-Attack, TextFooler, word-substitution attacks and defences |

> **Connection to this project:** The `AWP` class in `trainmodel.py` implements the same gradient-aligned weight perturbation strategy formalised in ART and CleverHans.  
> Applying these libraries to the trained model can quantify its robustness against adversarial social-media posts.

---

### 14.3 NLP-Based Threat & Anomaly Detection

| Repository | Description | Key Techniques |
|---|---|---|
| [elastic/detection-rules](https://github.com/elastic/detection-rules) | Production security-detection rules used by Elastic SIEM, including financial-sector threat patterns | KQL / EQL rules, MITRE ATT&CK mapping, ML jobs |
| [microsoft/presidio](https://github.com/microsoft/presidio) | Microsoft's PII & sensitive-data detection engine, widely deployed in banking compliance | Named-entity recognition, regex, transformer classifiers |
| [google/magika](https://github.com/google/magika) | Google's deep-learning content-type detector used in security pipelines | Fine-tuned transformer, multi-label classification |

---

### 14.4 Transformer Fine-Tuning Blueprints for Security Classification

| Repository | Description | Key Techniques |
|---|---|---|
| [huggingface/transformers](https://github.com/huggingface/transformers) | Reference implementations of DeBERTa, RoBERTa, BERT and every other backbone used in security classification | All major transformer architectures and fine-tuning recipes |
| [microsoft/DeBERTa](https://github.com/microsoft/DeBERTa) | Official DeBERTa repository (the backbone used in this project) | Disentangled attention, enhanced mask decoder |
| [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Semantic sentence embeddings used in banking KYC / compliance screening | Contrastive learning, bi-encoder, cross-encoder |

---

### 14.5 Key Takeaways for Banking Security Engineers

1. **Transformer backbones (DeBERTa, RoBERTa)** are the current state of the art for text-based fraud, phishing, and AML narrative classification — the same architecture used here.  
2. **Adversarial Weight Perturbation (AWP)** and adversarial training (ART, CleverHans) consistently improve robustness in security classifiers exposed to evasion attempts.  
3. **Label smoothing + multi-sample dropout** (used in this project) reduces overconfident predictions on borderline cases — critical in high-stakes banking decisions.  
4. **Graph Neural Networks** (AMLSim) complement text classifiers when transaction-network structure is available alongside message content.  
5. **Fairness-aware evaluation** (Feedzai benchmark) is essential for production banking systems to avoid disparate impact across demographic groups.
