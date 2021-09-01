---
title:  "XLNet: Generalized Autoregressive Pretraining for Language Understanding review"
toc: true
toc_sticky: true
permalink: /project/nlp/XLNet-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2021-08-17
---

## 들어가며

XLNet은 카네기 멜론 대학교와 구글 브레인에서 나온 논문으로, BERT가 bidirectionality를 얻는 과정에서 잃어버리는 정보를 만회하며 동시에 bidirectionality를 달성한다. 

## 1 Introduction

Unsupervised representation learning은 NLP 분야에서 엄청난 성공을 거둬왔다. 이러한 모델은 일반적으로 대량의 unlabeled text corpora로부터 pre-training을 통해 네트워크를 학습하고, downstream task에 대해  models/representations을 fine-tuning하게 된다. 이러한 고수준의 아이디어 하에 서로 다른 unsupervised pretraining objectives가 연구되어 왔다. 이 중 autoregressive (AR) language modeling와 autoencoding (AE)이 가장 큰 성공을 거둔 pretraining objective이다.

## 2 Proposed Method

### 2.1 Background

XLNet을 살펴보기 전에 전통적인 AR LM과 BERT에 대해 살펴보자.

**Auto Regressive Language Model**

주어진 text sequence $\mathbf x = [x _1, \cdots, x _T]$에 대해, AR language model은 forward autoregressive factorization에 대해 likelihood를 최대화하는 방식으로 pre-training을 진행한다. 즉, 다음과 같은 식이 된다.

$$
\max _\theta \log ~p _\theta (\mathbf x) = \sum^T _{t=1} \log ~p _\theta (x _t \rvert \mathbf x _{< t}) = \sum^T _{t=1} \log ~ \frac{\exp(h _\theta (\mathbf x _{1:t-1})^\intercal e(x _t))}{\sum _{x'} \exp (h _\theta (\mathbf x _{1:t-1})^\intercal e(x'))} \tag{1}
$$

여기서 $h _\theta (\mathbf x _{1:t-1})$는 RNN/Transformer와 같은 네트워크에 의해 생성되는 context representation이고, $e(x)$는 $x$의 임베딩이 된다. 

(아무래도 $\max$가 각 식마다 들어가야 할 것 같지만, 우선은 논문에 소개된대로 적는다.)

**BERT (denoising auto encoding)**

반면 BERT는 주어진 text sequence $\mathbf x = [x _1, \cdots, x _T]$에 대해 임의의 단어를 일정 확률 (15%)로 `[MASK]` 토큰으로 변경하여 오염된 버전의 sequence $\hat {\mathbf x}$를 생선한다. 마스킹된 토큰을 $\bar{\mathbf x}$라 하자. Training objective는  $\hat{\mathbf x}$으로부터 $\bar{\mathbf x}$를 복구하는 것이 된다.

$$
\max _\theta \log ~p _\theta (\hat{\mathbf x} \rvert \bar{\mathbf x}) 
\approx \sum^T _{t=1} m _t \log ~p _\theta (x _t \rvert \hat{\mathbf x}) 
= \sum^T _{t=1} m _t \log ~ \frac{\exp(H _\theta (\hat{\mathbf x} _t)^\intercal e(x _t))}{\sum _{x'} \exp (H _\theta (\hat{\mathbf x _t} _t)^\intercal e(x'))} \tag{2}
$$

$m _t=1$은 $x _t$가 마스킹되었음을 뜻하고, $H _\theta$는 Transformer로, $T$길이의 text sequence $\mathbf x$를 hidden vector의 sequence $H _\theta(\mathbf x) = [H _\theta(\mathbf x) _1, H _\theta(\mathbf x) _2, \cdots, H _\theta(\mathbf x) _T]$로 변환하는 역할을 한다.

이들의 장단점은 다음과 같다.

**Independence Assumption**

식 (2)을 자세히 보면 같다($=$)가 아니고 근사한다($\approx$)이다. BERT는 joint conditional probability $p (\hat{\mathbf x} \rvert \bar{\mathbf x})$를 factorize하는데, 이는 마스크된 모든 토큰이 **각자 복원**된다는 뜻이다. 즉, 마스크된 토큰끼리는 **서로 독립적**이라는 가정하에 복원을 진행하게 된다.

예를들어 *New York City*라는 단어가 있다고 가정하자. 우리는 *New*와 *York*가 긴밀한 관계에 있음을 알고있다. 그러나 BERT는 이런 관계를 무시하고(독립적) 복원을 하게 된다.

이에 대한 예시는 Appendix 5.1 Comparison with BERT에 잘 나와있다.

**Input noise**

BERT는 인공적인 심볼 `[MASK]`를 통해 pre-training을 진행하게 되는데, 이는 downstream task에서는 **절대 등장하지 않아** pre-training과 fine-tuning사이의 **불일치**를 만들게 된다. 이를 막기위해서 BERT는 마스크된 토큰 중 80%는 그대로, 10%는 랜덤하게, 10%는 원래 단어로 다시 되돌리게 되는데, 이 확률자체가 너무 작기 때문에 이러한 불일치 문제를 해결하기가 어렵다. 그렇다고 이 확률을 키우게되면 최적화하기 trivial한 문제가 된다. AR 모델은 다행히 이러한 문제를 겪지 않는다.

**Context dependency**

AR은 왼쪽방향에 있는 토큰에만 의지하게된다. 이는 **bidirectional context를 잡지못하는 문제**가 발생한다.

### 2.2 Objective: Permutation Language Modeling

XLNet은 permutation language modeling objective를 통해 AR의 장점은 유지하면서 bidirectional context도 잡을 수 있게한다. 길이 $T$를 갖는 sequence $\mathbf x$는 $T!$의 순서를 갖을 수 있다. 이러한 factorization order에 상관없이 모델을 학습시키면 양쪽 모두의 방향에서의 정보를 얻을 수 있을 것이다.

이러한 아이디어를 정리해보자. $T$길이의 index sequence $[1, \cdots, T]$의 모든 permutation 집합을 $\mathcal Z _T$라 해보자. 어떤 permutation $\mathbf z \in \mathcal Z$의 $t$번째 element를 $z _t$, $t-1$까지의 element들을 $\mathbf z _{<t}$라 하자. 그러면 permutation language modeling objective는 다음과 같이 적을 수 있다.

$$
\max \mathbb E _{\mathbf z \sim \mathcal Z _T} \left [ \sum ^T _{t=1} \log p _\theta (x _{z _t} \rvert \mathbf x _{\mathbf z _{<t}})  \right ] \tag{3}
$$





\mathbf x

{: .align-center}{: width="700"}

{: .text-center}
