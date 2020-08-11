---
title:  "Beam Search (아직 작성 중)"
excerpt: "Beam Search에 대한 이해와 구현"
toc: true
toc_sticky: true

categories:
  - PyTorch
  - NLP

use_math: true
last_modified_at: 2020-08-12
---

## Introduction

[Transformer 구현](https://inhyeokyoo.github.io/pytorch/nlp/NLP-Transformer-Impl-Issues/)을 하던 도중 Beam Search에 대한 이해가 부족한 것 같아 정리를 해보려 한다.
논문의 링크는 [여기](https://arxiv.org/pdf/1606.02960.pdf)를, reference는 따로 밑에다 달도록 하겠다.

## Background and Notation

시작하기 앞서 논문 3장을 보면서 notation을 익혀보자

- 일단 input sequence가 인코딩되면, seq2seq은 *디코더*를 사용하여 target vocabulary $\mathcal V$로부터 target sequence를 생성
    - 특히, 이 생성되는 단어들은 input representation $x$와 이전에 생성된 단어들이나 *history*에 조건부로 생성
- 여기서 우리는 $w_{1:T}$라는 notation을 사용하여 T길이의 임의의 sequence를 표현할 것임
- $y_{1:T}$는 x에 대한 *gold*(정답) target word sequence를 의미함
- $m_1, ..., m_T$를 sequence of T vector로, $h_0$를 initial state vector로 하자
- 이러면 RNN에 어떠한 sequence를 적용하더라도 다음과 같은 $h_t$를 생성한다
    - $h_t \leftarrow \textrm{RNN}(m_t, h_{t-1}; \theta)$
- 여기서 $m_t$는 항상 target word sequence $w_{1:T}$에 대응하는 embedding
- 따라서 다음과 같이 써도 무방함
    - $h_t \leftarrow \textrm{RNN}(w_t, h_{t-1}; \theta)$
    - 여기서 $w_t$는 항상 이의 embedding을 의미함
- $p(w_t|w_{1:t-1}, x) = g(w_t, h_{t-1}$ $\bm