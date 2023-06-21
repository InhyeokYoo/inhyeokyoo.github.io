---
title:  "QA task와 데이터셋을 알아보자"
toc: true
toc_sticky: true
permalink: /project/nlp/Retrieval-benchmark/
categories:
  - NLP
tags:
  - benchmark
use_math: true
last_modified_at: 2023-06-21
---

## 들어가며


MS MARCO passage ranking dataset
- widely-used ad-hoc retrieval benchmark with 8.8 millions passages.
- The training set contains 0.5 million pairs of queries and relevant passages, where each query on average has one relevant passage
- https://microsoft.github.io/msmarco/.

MS MARCO Dev Queries
- MS MARCO dataset’s official dev set, which has been widely used in prior research [28,8]. 
- It has 6,980 queries. Most of the queries have only 1 document judged relevant; the labels are binary.

TREC2019 DL Queries
- the official evaluation query set used in the TREC 2019 Deep Learning Track shared task
- It contains 43 queries that are manually judged by NIST assessors with 4-level relevance labels, allowing us to understand the models’ behavior on queries with _multiple_, _graded relevance judgments_ (on average 94 relevant documents per query).

- BERT-Siamese performs worse than BM25 in terms of MAP@1k and recall on TREC DL queries.
-> TREC DL query has multiple relevant documents with graded
relevance levels. It therefore requires a better-structured embedding space to
capture this, which proves to be harder to learn here. clear circumvents this
full embedding space learning problem by grounding in the lexical retrieval model
while using embedding as complement.


{: .text-center}
