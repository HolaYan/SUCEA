# SUCEA: Reasoning-Intensive Retrieval for Adversarial Fact-checking through Claim Decomposition and Editing
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXX-blue)](link_to_arxiv) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"SUCEA: Reasoning-Intensive Retrieval for Adversarial Fact-checking through Claim Decomposition and Editing"**.

## 📋 Abstract
Automatic fact-checking has recently received more attention as a means of combating misinformation. Despite significant advancements, fact-checking systems based on retrieval-augmented language models still struggle to tackle adversarial claims, which are intentionally designed by humans to challenge fact-checking systems. To address these challenges, we propose a training-free method designed to rephrase the original claim, making it easier to locate supporting evidence. Our modular framework, SUCEA, decomposes the task into three steps: 1) Claim Segmentation and Decontextualization that segments adversarial claims into independent sub-claims; 2) Iterative Evidence Retrieval and Claim Editing that iteratively retrieves evidence and edits the subclaim based on the retrieved evidence; 3) Evidence Aggregation and Label Prediction that aggregates all retrieved evidence and predicts the entailment label. Experiments on two challenging fact-checking datasets demonstrate that our framework significantly improves on both retrieval and entailment label accuracy, outperforming four strong claim-decomposition-based baselines.

## 🧠 Highlights
- 🔍 What problem you solve (e.g., "first to apply method X to task Y")
- 💡 Key novelty (e.g., "multi-modal attention fusion with implicit grounding")
- 🚀 Performance (e.g., "achieves SOTA on dataset Z")

## 🗂️ Project Structure
```bash
.
├── Parsing/                  
├── text_paraphrase/               
├── eval/               
├── fact_checking/              
├── multi_round_retriever/            
└── README.md             
