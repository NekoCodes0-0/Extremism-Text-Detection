# Extremism Detection in Social Media Text
## Top-10 Solution in a Kaggle Competition

---

## Abstract

This repository presents a complete deep learning pipeline for detecting extremist content in social media text.  
The task is formulated as a binary classification problem distinguishing between extremist and non-extremist messages.

The solution was developed for a Kaggle competition on digital extremism detection and achieved a **Top-10 position on the public leaderboard**.  
The approach focuses on contextual understanding, robustness to noisy language, and stable generalization rather than keyword-based heuristics.

---

## Table of Contents

1. Introduction  
2. Problem Definition  
3. Challenges in Extremism Detection  
4. Methodology  
5. Model Architecture  
6. Text Preprocessing  
7. Training Strategy  
8. Inference Pipeline  
9. Experimental Results  
10. Repository Structure  
11. Setup and Usage  
12. Limitations  
13. Conclusion    

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
👉The model may struggle with extremely short or context-free messages
👉Cultural and regional slang not present in training data can affect accuracy
👉Binary classification does not capture degrees or categories of extremism

## 13. Conclusion

This project demonstrates that robust extremism detection requires more than keyword filtering.
Contextual modeling, regularization, and disciplined training strategies are essential for handling noisy, real-world social media text.

The results validate the effectiveness of combining large pretrained transformers with stability-focused enhancements.
