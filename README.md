# Code Switch Prediction

## Overview
This project investigates early code-switch prediction directly from raw speech, without relying on transcripts or speech-to-text systems. It examines whether self-supervised speech representations contain information about an upcoming code-switch, which representation layers carry the strongest early-warning signal, and how far in advance a code-switch can be reliably predicted.

## Data
The ASCEND (A Spontaneous Chinese-English Dataset for Code-switching in Multi-turn Conversation) dataset is used. 

## Methodology
### 1. Ground truth construction using MMS_FA and manual verification of a representative subset

### 2. Feasibility study 
- Language identity 
    Raw audio -> 20ms SSL layer representation 
    Each layer -> logistic regression -> EN/ZH
    Each layer -> UMAP
- Pre-switch information  
    Raw audio -> SSL layer representation 
    Each layer -> logistic regression -> normal/pre-switch (normal: >500ms; pre-switch: <=500ms from switch)
    Each layer -> UMAP

### 3. Proposed methodology
Raw speech -> SSL representations -> try all layers -> context windows -> temporal predictor -> p(switch within ∆ms | context)

### 4. Evaluation metrics
- ROC-AUC
- PR-AUC 
- Log loss / binary cross entropy
- Brier score
- Expected Calibration Error (ECE)
- Reliability diagram
- Lead time

### 5. Baseline
- Base rate 
- eGeMAPSv02 LLDs

### 6. Comparative experiments
- SSL model (wav2vec 2.0, HuBERT, WavLM, XLS-R)
- SSL layer
- Temporal context 
- Temporal predictor (LSTM, TCN, Transformer)

### 7. Ablation study 
- Without temporal context, using only the current 20ms frame-level representation

## Results