# SUCEA: Reasoning-Intensive Retrieval for Adversarial Fact-checking through Claim Decomposition and Editing
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXX-blue)](link_to_arxiv) 
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
| Source           | Wikipedia        | Wikipedia        |
| # Claims (Test)  | 200              | 358              |
| Claim Length     | 14.03 tokens     | 24.20 tokens     |
| Avg # Evidence   | 1.29             | 3.94             |

## 🗂️ Project Structure
```bash
.
├── Parsing/ # Claim segmentation and decontextualization
├── text_paraphrase/ # Evidence-guided paraphrasing
├── eval/ # Evaluation metrics and ablation tools
├── fact_checking/ # Final label prediction and evidence aggregation
├── multi_round_retriever/ # Retrieval loop with first- and second-round logic
└── README.md

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
@article{liu2024sucea,
  title={SUCEA: Reasoning-Intensive Retrieval for Adversarial Fact-checking through Claim Decomposition and Editing},
  author={Liu, Hongjun and Zhao, Yilun and Cohan, Arman and Zhao, Chen},
  journal={arXiv preprint arXiv:2407.21783},
  year={2024}
}
