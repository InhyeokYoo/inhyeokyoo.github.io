---
title:  "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter review"
toc: true
toc_sticky: true
permalink: /project/nlp/DistilBERT-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2022-05-22
---

## Introduction

이번 포스트에서 살펴볼 논문은 무려 NeurIPS에서 발표한 DistillBERT이다.
Hugging Face에서 발표하였으며, 

Accepted at the 5th Workshop on Energy Efficient Machine Learning and Cognitive Computing - NeurIPS 2019

이번부터는 포스트 형태를 조금 바꾸기로 하였다.
그냥 논문만 번역하는 느낌이 들어 내가 이해한 것을 더욱 적극적으로 적는 형태로 바뀌었으니, 자세한 내용은 원본을 참조하길 바란다.

## Challenges

매번 논문마다 비슷한 이야기를 하는 것 같지만, 결국 본 논문의 핵심은 모델 사이즈를 어떻게 줄이느냐이다.
NLP에서 표준이 된 Transfer Learning은 파라미터가 많을수록 더 좋은 성능을 내지만, 다음과 같은 문제점이 있다.

1. 모델이 커짐에 따라 연산 비용도 급격하게 증가한다.
2. 리얼타임으로 서버가 아닌 온디바이스(on-device)로 동작하기가 어렵다.

결국 지금까지 포스팅한 연구의 큰 갈래를 보자면 Transfer learning이 트렌디하다보니 이에 대한 loss/archtecture 연구가 발전하였고 (e.g. XLNet, MASS, BART), scaling 모델로 성능을 향상 (e.g. RoBERTa, GPT-3), 그리고 경량화연구로 (e.g. DistillBERT, ALBERT) 진행되는 것으로 보인다.

## Contributions

DistillBERT의 contribution은 다음과 같다.

- **knowledge distillation**을 통해 학습시킨 LM은 성능도 좋으면서 파라미터도 적다
- 저자들의 general-purpose pre-trained model은 몇몇 downstream task에서 좋은 성능을 갖도록 fine-tune할 수 있으며, 큰 모델의 유연성까지 유지할 수 있다.
- triple loss를 통해 40% 가볍고 60%빠른 모델 만듬.

저자들은 모델이 가볍기 때문에 압축한 모델을 통해 모바일과 같은 환경에서도 동작할 수 있다고 한다.

## Background: Knowledge distillation

Knowledge distillation 쪽은 가볍게 알기때문에 조금 꼼꼼하게 살펴보겠다.

그 유명한 제프리 힌튼의 Knowledge distillation은 큰 모델이나 앙상블 모델 (teacher)의 동작방식을 가벼운 모델 (student)에 학습시켜 이를 재생성하는 모델 압축 기술을 의미한다.

일반적으로 supervised learning에서는 분류 모델이 레이블에 대한 확률의 추정값을 최대화하여 데이터의 클래스를 예측하는 방식으로 학습이 이루어진다.







## Method

most of the operations used in the Transformer architecture (linear
layer and layer normalisation) are highly optimized in modern linear algebra frameworks and our
investigations showed that variations on the last dimension of the tensor (hidden size dimension) have
a smaller impact on computation efficiency (for a fixed parameters budget) than variations on other
factors like the number of layers. Thus we focus on reducing the number of layers.

taking advantage of the common dimensionality between teacher and student networks,
we initialize the student from the teacher by taking one layer out of two.

Pooler 관련:
https://discuss.huggingface.co/t/what-is-the-purpose-of-the-additional-dense-layer-in-classification-heads/526/3

## Experiment


{: .align-center}{: width="600"}