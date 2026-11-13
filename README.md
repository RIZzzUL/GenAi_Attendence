# Retrieval-Augmented Generation (RAG): Implementation

Individual research paper implementation for **Generative AI (2311312T)**.

**Student:** Rijul Samaiya
**Program:** B.Tech CSE (AIML), MIT Academy of Engineering, Pune

## Paper

**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**
Patrick Lewis et al., NeurIPS 2020
[arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

## Overview

Large language models store knowledge in their weights, which goes stale, cannot be traced to a source, and can be hallucinated. RAG addresses this by combining:

- a **retriever** that finds relevant passages from an external knowledge base, and
- a **generator** that produces the answer conditioned on those passages.

This project implements the core RAG methodology and compares it against a generator that has no retrieval.

## Implementation

| Component | Choice |
|---|---|
| Dataset | SQuAD (validation subset: 300 QA pairs, 100 evaluated) |
| Knowledge base | Unique SQuAD passages from the subset |
| Retriever | `all-MiniLM-L6-v2` embeddings + FAISS (`IndexFlatIP`, cosine similarity) |
| Generator | `google/flan-t5-base` |
| Baseline | Same generator without retrieved context |
| Metrics | Exact Match, F1, Retrieval Recall@k |
| Ablation | Number of retrieved passages, k = 1, 3, 5 |

### Pipeline

```
Question -> embed -> FAISS top-k search -> retrieved passages
         -> prompt (context + question) -> Flan-T5 -> answer
```

## Results

| Model | k | Recall@k | EM | F1 |
|---|---|---|---|---|
| No-RAG | - | - | `<fill>` | `<fill>` |
| RAG | 1 | `<fill>` | `<fill>` | `<fill>` |
| RAG | 3 | `<fill>` | `<fill>` | `<fill>` |
| RAG | 5 | `<fill>` | `<fill>` | `<fill>` |

`<Add one or two sentences summarising the main finding.>`

## Differences from the original paper

1. The retriever is frozen (no end-to-end training).
2. Retrieved passages are concatenated into the prompt, rather than marginalized as in RAG-Sequence / RAG-Token.
3. A small SQuAD-based corpus is used instead of Wikipedia.
4. The generator is Flan-T5 instead of BART.

## Limitations and future improvements

- Retrieval errors lead to wrong answers: use hybrid BM25 + dense retrieval and a cross-encoder re-ranker.
- Fixed passages and no chunking strategy: use overlapping chunks.
- No marginalization over documents: implement RAG-Sequence scoring.
- Frozen retriever: fine-tune it on QA pairs.
- EM is strict: add semantic metrics such as BERTScore.

## How to run

1. Open the notebook in Google Colab:
   [Open in Colab](https://colab.research.google.com/github/RIZzzUL/GenAi_Attendence/blob/main/<your_notebook_name>.ipynb)
2. Select *Runtime -> Run all* (a T4 GPU is optional, CPU also works).

## Repository contents

- `<your_notebook_name>.ipynb`: executable Colab notebook
- `RAG_Lewis_et_al_2020.pdf`: selected research paper
- `<your_report_name>.pdf`: implementation report
- `screenshots/`: screenshots of the implementation and results

## Reference

Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
