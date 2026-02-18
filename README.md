# LLM Reliability with Context

## Overview

This project investigates how contextual reliability affects the correctness and uncertainty of language model outputs.

Rather than attempting to eliminate hallucinations, the goal is to measure whether simple uncertainty signals (specifically token-level entropy) correlate with incorrect answers when context is degraded.

The model is treated as a black box and evaluated under controlled experimental conditions.

---

## Motivation

Large language models frequently produce fluent but incorrect answers.  
In many real-world systems, detecting unreliable outputs is more important than maximizing raw accuracy.

This project explores:

- How accuracy changes when context quality degrades
- Whether entropy can act as a lightweight hallucination risk signal
- Whether misleading context is more harmful than missing context

---

## Experimental Design

Each factual question is evaluated under three conditions:

### 1. Clean Context
Correct and supportive factual information is provided.

### 2. Corrupted Context
Context contains plausible but incorrect information designed to mislead the model.

### 3. Missing Context
No supporting information is provided; the model relies purely on prior knowledge.

The questions remain identical across conditions to isolate context reliability as the only variable.

---

## Methodology

- Model: `distilgpt2` (open-source transformer)
- Stochastic sampling enabled
- Multiple generations per prompt
- Correctness measured via soft token-based matching
- Entropy computed from first-token probability distribution

Metrics recorded:

- Binary correctness (0 / 1)
- Token-level entropy
- Condition label (clean / corrupted / missing)

---

## Results Summary

Observed trends:

- Accuracy: Clean > Corrupted > Missing
- Entropy: Clean < Corrupted ≈ Missing
- Higher entropy correlates strongly with incorrect responses

Key insight:

Misleading context reduces accuracy more significantly than missing context, and elevated entropy serves as a useful indicator of unreliable outputs.

---

## Example Plots

- Accuracy by condition
- Average entropy by condition
- Entropy vs correctness scatter plot

(See generated plot files in repository.)


