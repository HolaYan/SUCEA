# SUCEA: Reasoning-Intensive Retrieval for Adversarial Fact-checking through Claim Decomposition and Editing
[![arXiv](https://img.shields.io/badge/arXiv-2506.04583-blue)](https://arxiv.org/abs/2506.04583)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"SUCEA: Reasoning-Intensive Retrieval for Adversarial Fact-checking through Claim Decomposition and Editing"**.

## 📋 Abstract
Automatic fact-checking has recently received more attention as a means of combating misinformation. Despite significant advancements, fact-checking systems based on retrieval-augmented language models still struggle to tackle adversarial claims, which are intentionally designed by humans to challenge fact-checking systems. To address these challenges, we propose a training-free method designed to rephrase the original claim, making it easier to locate supporting evidence. Our modular framework, SUCEA, decomposes the task into three steps: 1) Claim Segmentation and Decontextualization that segments adversarial claims into independent sub-claims; 2) Iterative Evidence Retrieval and Claim Editing that iteratively retrieves evidence and edits the subclaim based on the retrieved evidence; 3) Evidence Aggregation and Label Prediction that aggregates all retrieved evidence and predicts the entailment label. Experiments on two challenging fact-checking datasets demonstrate that our framework significantly improves on both retrieval and entailment label accuracy, outperforming four strong claim-decomposition-based baselines.

## 🧠 Key Contributions

- ✂️ **Claim Segmentation and Decontextualization**: Converts complex claims into atomic, context-independent subclaims using LLMs.
- 🔁 **Iterative Retrieval + Evidence-Guided Claim Editing**: Uses a two-round retrieval-paraphrase loop to refine claims based on real evidence, enhancing retrievability.
- 🧩 **Evidence Aggregation and Final Prediction**: Aggregates all retrieved passages and prompts LLMs for final entailment label.

## 🚀 Performance Highlights

- 📈 **+7.5% Fact-checking Accuracy** on FOOLMETWICE and **+5.9%** on WICE over RALM baseline.
- 🔍 **+11% Retrieval Recall@10** under TF-IDF using SUCEA’s editing module.
- 🔄 Robust across LLMs: Works effectively with GPT-4o-mini, LLaMA-3.1-70B, Mistral 7B, and others.

## 📊 Datasets

| Property          | FOOLMETWICE     | WICE             |
|------------------|------------------|------------------|
| Corpus           | Wikipedia        | Wikipedia        |
| # Claims (Test)  | 200              | 358              |
| Claim Length     | 14.03 words     | 24.20 words     |
| Avg # Evidence   | 1.29             | 3.94             |

## 📚 Knowledge Corpus

We use the **December 20, 2018 Wikipedia dump** as the external retrieval corpus. To enhance retrieval performance and ensure consistency across methods, the corpus is preprocessed into fixed-length passages.

- 📦 **Corpus Size**: 21,015,325 passages
- 📄 **Passage Format**: Each passage contains **100 tokens**.
- 🧹 **Preprocessing**:
  - Wikipedia articles are tokenized and split into non-overlapping windows.
  - Each window is treated as an independent passage for retrieval.
  - Stopword removal and basic normalization are applied for TF-IDF.
- 🔍 **Used in**:
  - **First-Round Evidence Retrieval** (on original subclaims)
  - **Second-Round Retrieval** (on paraphrased subclaims)
  - Supports both **TF-IDF** and **Contriever** retrievers

We provide a script to process the Wikipedia dump into this passage format (see `multi_round_retriever/preprocess_wiki.py`).

## 🗂️ Project Structure
```bash
.
├── Parsing/ # Claim segmentation and decontextualization
├── text_paraphrase/ # Evidence-guided paraphrasing
├── eval/ # Evaluation metrics and ablation tools
├── fact_checking/ # Final label prediction and evidence aggregation
├── multi_round_retriever/ # Retrieval loop with first- and second-round logic
└── README.md
```

## 🧪 Getting Started

Coming soon: Instructions for reproducing the results in the paper using GPT-4o-mini or LLaMA-3.

## 📈 Results Summary

| System     | FOOLMETWICE | WICE |
|------------|-------------|------|
| RALM       | 65.5        | 31.2 |
| MINICHECK  | 70.5        | 35.2 |
| SUCEA      | **73.5**    | **38.7** |

## 📚 Citation
```bibtex
@misc{liu2025suceareasoningintensiveretrievaladversarial,
      title={SUCEA: Reasoning-Intensive Retrieval for Adversarial Fact-checking through Claim Decomposition and Editing}, 
      author={Hongjun Liu and Yilun Zhao and Arman Cohan and Chen Zhao},
      year={2025},
      eprint={2506.04583},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.04583}, 
}
