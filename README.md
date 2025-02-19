# RAG-Generated Augmented Datasets for Document Rank Boosting in Information Retrieval
![](/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval/images/arch.png)
<p align="center"><em>Model Architecture.</em></p>

## Overview

This repository contains the code, datasets, and experimental results associated with our SIGIR 2025 paper:

"RAG-Based Datasets for Adversarial Rank Attacks Against Neural Rankers"

In this paper, we explore adversarial rank attacks on Neural Ranking Models (NRMs) using a structured framework based on Retrieval-Augmented Generation (RAG). Our approach generates adversarial datasets by strategically modifying documents with adversarially generated sentences to manipulate ranking results in Information Retrieval (IR) systems.

## Paper Abstract

Neural Ranking Models (NRMs) are vulnerable to adversarial attacks that manipulate document rankings, threatening the integrity of retrieval systems. Existing adversarial attacks rely on unsupervised methods and surrogate models, which limit their generalizability. In this work, we introduce a novel RAG-based dataset construction framework that employs Large Language Models (LLMs) to generate adversarially modified documents optimized for rank manipulation. Our dataset is released in three variations: Gold, Platinum, and Diamond, categorized by attack effectiveness. The proposed approach facilitates robust evaluation of adversarial resilience in IR systems and the development of defense strategies against ranking attacks.

## Key Contributions

- A supervised adversarial dataset for benchmarking adversarial attacks against NRMs.
- An iterative self-refinement process using an LLM-NRM feedback loop to generate high-impact adversarial modifications.
- Three dataset variations (Gold, Platinum, Diamond) categorized by ranking impact.
- Comprehensive evaluation against state-of-the-art adversarial attack methods on IR systems.

## Installation
To set up the project environment, run the following:

```
conda create -n rag_attack_env python=3.9
conda activate rag_attack_env
pip install -r requirements.txt
```

## Datasets
Our datasets are categorized as follows:
- Gold: Includes all adversarially modified documents with at least one rank improvement.
- Platinum: Selects the best-performing adversarial modifications per document-query pair.
- Diamond: Strictly filters cases where documents achieve a rank of ≤10 for mid-ranked (Easy-5) and ≤50 for lower-ranked (Hard-5) documents.

## Experimental Results
We evaluate our approach against state-of-the-art adversarial ranking techniques, including:
- Query Injection (Query+)
- Embedding Perturbation (EMPRA)
- Trigger-Based Attacks (PAT, Brittle-BERT)
- Sentence-Level Attacks (IDEM)

Our RAG-based approach achieves a 100% attack success rate, significantly improving document rankings while maintaining high linguistic fluency and naturalness.