---
title:  "Deep contextualized word representations 정리"
excerpt: "ELMO 정리"
toc: true
toc_sticky: true

categories:
  - NLP
tags:
  - Paper

use_math: true
last_modified_at: 2020-08-31
---

# Intro.

NLP 스터디에서 ELMo를 공부할 차례가 되서 간략하게 정리해보았다. 원 논문은 [다음](https://arxiv.org/abs/1802.05365)에서 확인할 수 있다.

# 3. ELMo: Embeddings from Language Models

널리 사용되는 대부분의 word embeddings과는 다르게, ELMo word representations는 input sentence 전체에 대한 functions이다. ELMo word representations는 character convolutions와 두 개의 biLMs의 맨 윗 부분에서 계산되고, internal netowkr states라는 linear function를 통과한다. 이러한 setup은 biLM 이 large scale에서 pre-train되고 현존하는 넓은 범위의 neural NLP architecture를 쉽게 통합할 수 있는 곳에서 semi-supervised learning을 가능케한다.

## 3.1. Bidirectional language models

N tokens의 sequence가 주어질 때 ($t _1, t _2, ..., t _N$), forward language model은 이 sequence의 확률을 계산한다. 이는 token given history $t_1, t_2, ..., t _{k-1}$이 주어질 때, $t _k$의 확률을 모델링함으로 얻을 수 있다.  

$$
\begin{align}
p(t _1, t _2, ..., t _N) = \Pi^{N} _{k=1} p(t _k \lvert t _1, t _2, ..., t _{k-1})
\end{align}
$$

최근 SOTA Neural LM은 문맥(context)에 독립적인 토큰 표현(token-representation) $x^{LM} _k$를 토큰 임베딩 혹은 characters에 대한 CNN을 통해 계산하고, 그 후 정방향(forward) LSTM의 L개의 레이어를 통과한다. 
각 position $k$에서는 각 LSTM 레이어가 문맥(context)에 의존적인(dependent) 표현(representation) $\overrightarrow {h^{LM} _{k, j}}$를 계산한다. 여기서 $j=1, ..., L$이다.
맨 위 레이어(The top layer) LSTM의 결과 $\vec {h^{LM} _{k, j}}$는 Softmax layer를 통해 다음 토큰 $t _{k+1}$을 예측하는데 사용된다.

역방향(backward) LM은 sequence를 반대순서로 돌고, 앞선 시점의 문맥을 통해 이전 토큰을 예측한다는 점만 빼면 정방향 LM과 비슷하다.  

$$
\begin{align}
p(t _1, t _2, ..., t _N) = \Pi^{N} _{k=1} p(t _k \lvert t _{k+1}, t _{k+2}, ..., t _{N})
\end{align}
$$

이는 정방향 LM과 유사한 방법으로 구현할 수 있으며, 각 역방향 LSTM 레이어 j는 $t _{k+1}, t _{k+2}, ..., t _{N}$가 주어질 때 $t _k$의 표현(representation) $\overleftarrow {h^{LM} _{k, j}}$을 계산한다.

BiLM은 정방향과 역방향 LM 두개 모두를 합친다. 논문의 공식화(formulation)은 정방향과 역방향의 log likelihood를 동시에 최대화한다.  

$$
\begin{align}
\sum^N _{k=1} (\log p(t _k \lvert t_1, ..., t _{k-1}; \Theta _x, \overrightarrow \Theta _{LSTM}, \Theta _s) \\
+(\log p(t _k \lvert t _{k+1}, ..., t _{N}; \Theta _x, \leftrightarrow \Theta _{LSTM}, \Theta _s))
\end{align}
$$

정방향과 역방향에 모두에서 토큰 표현 ($\Theta _x$)과 Softmax layer ($\Theta _s$)를 위한 파라미터를 통일시키고, LSTM의 각 방향을 위한 parameter ($\overrightarrow \Theta _{LSTM}, \leftrightarrow \Theta _{LSTM}$)는 분리를 유지한다. 이 공식화는 완벽하게 독립적인 파라미터 대신 방향사이의 weight를 공유한다는 점을 제외하면 전체적으로 Peters et al. (2017)과 비슷하다.

## 3.2. EMLo

ELMo는 biLM의 중간 레이어의 표현의 *task-specific*한 조합이다. 각 토큰 $t _k$에 대해, *L*-layer biLM은 $2L+1$개의 표현을 배운다.  

$$
\begin{align}
R _k &= \{ \mathbf{x}^{LM} _k, \overrightarrow {\mathbf{h}^{LM} _{k, j}}, \overleftarrow {\mathbf{h}^{LM} _{k, j}} \lvert j=1, ..., L\} \\
&= \{ \mathbf{h}^{LM} _{k, j} \lvert j=0, ..., L \},
\end{align}
$$

각 biLSTM 레이어에 대해 $\mathbf{h}^{LM} _{k, 0}$는 토큰 레이어이고, $\mathbf{h}^{LM} _{k, j} = [\overrightarrow {\mathbf{h}^{LM} _{k, j}} ; \overleftarrow {\mathbf{h}^{LM} _{k, j}}]$이다.

다운스트림 모델에 포함하기 위해, ELMo는 *R*에 있는 모든 레이어를 하나의 벡터($\textrm{ELMo} _k = E(R _k; \mathbf{\Theta} _e$)로 붕괴(collapse)한다. 가장 간단한 케이스에서는, ELMo는 TagLM과 CoVe에서 처럼 맨 위의 레이어를 선택($E(R _k)=\mathbf{h}^{LM} _{k, L}$)한다. 좀 더 일반적으로, 모든 biLM 레이어의 task specific한 웨이트를 계산한다: