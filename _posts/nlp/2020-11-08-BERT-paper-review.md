---
title:  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding review (작성 중)"
excerpt: "BERT Paper review"
toc: true
toc_sticky: true
permalink: /project/nlp/bert-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling

use_math: true
last_modified_at: 2020-11-08
---

이번 시간엔 시대를 풍미했던 SOTA모델 BERT에 대해 알아보자. 원문은 [다음 링크](https://arxiv.org/abs/1810.04805)에서 확인할 수 있다.

# 0. Abstract

- 본 논문에선 새로운 language representation BERT (**B**idirectional **E**ncoder **R**epresentations  from **T**ransformers)를 소개 
    - 최근의 language representation (GPT-1, ELMO) 와는 다르게, BERT는 left/right context 모두를 unlabeled data로부터 동시에 (jointly conditioning) 사전학습 (pre-train) 할 수 있도록 설계되
    - 이 결과로 단지 하나의 output layer를 추가하여 fine-tuning 하는 것만으로 다양한 downstream task (Q.A, language inference)에서 SOTA를 달성할 수 있었다.
- 자세한 결과로는
    - GLUE score는 80.5%
    - MultiNLI는 86.7%의 정확도
    - SQuAD v1.1 QA는 F1 test를 93.2
    - SQuAD v2.0 QA는 F1 test를 83.1

# 1. Introduction

- Language modeling pre-training은 다양한 NLP task에서 효과적임을 보임
    - GPT-1
    - ELMo
    - *Semi-supervised sequence learning*
    - *ULMFiT*
- 이에는 sentence-level task (e.g. language inference), paraphrasing, QA, NER 등이 있음

*Semi-supervised sequence learning*: 이 논문은 2015년도에 나온 논문인데, unlabeld data를 이용하여 sequence learning을 진행한 논문이다. 일반적인 LM ($p(x _t \rvert x _1, ..., x _{t-1})$)과 sequence autoencoder를 써서 괜찮은 결과를 뽑아냈다는 논문이다. 지금와서 생각해보면 그냥 그런거 같은데, 2015년에 나온걸 감안하면 세대를 앞서갔다는 느낌도 든다. 원문은 [다음](https://arxiv.org/pdf/1511.01432.pdf)을 참고.
{: .notice--info}

*ULMFiT*: 앞선 GPT-1에서 보았던 discriminative fine-tuning을 제안한 논문이다. 이 논문은 general-domain corpus에 대해 pre-train하고, target task에 대한 data에 대해 discriminative fine-tuning과 slanted triangular learning rates를 이용하여 fine-tuning을 진행한 뒤, classifier를 gradual unfreezing, discriminative fine-tuning, slanted triangular learning rates를 이용하며 fine-tuning한다.
{: .notice--info}

- pre-trained language representation을 downstream task에 적용하는 것은 두 가지가 있음
- (1). feature-based
    - pre-trained representation을 포함하는 task-specific architecture를 추가적인 feature로 이용하는 것 
    - ELMo가 그 예시인데, ELMo에서 LM representation과 embedding을 concat해서 사용하는 것을 생각하면 된다.
- (2). fine-tuning approach
    - 최소한의 task-specific parameter만을 도입하고, downstream에 대해 모든 parameter를 그냥 fine-tuning하는 것
    - GPT-1이 대표적
- 이러한 두 가지 방법은 pre-training에서 같은 objective function (아마도 $p(x _t \rvert x _1, ..., x _{t-1})$)을 공유하고, unidirectional language model을 통해 일반적인 language representation을 학습한다.

- 본 논문에서는 이러한 방법이 pre-trained representation의 능력을, 특히 fine-tuning 방법에서, **제약**시킨다고 본다.
    - 가장 큰 제약은 일반 LM이 **단방향**이라는 것이고, 이는 pre-training에서 사용할 수 있는 **모델의 선택지를 제한**한다
    - 예를 들어 GPT의 경우엔 left-to-right architecture (transformer decoder)를 사용하였는데, 이는 subsquence mask로 인해 오직 **이전의 token**에만 self-attnetion이 가능하다.
    - 이러한 제약은 sentence-level에선 *sub-optimal*이고, token-level task (e.g. QA)와 같이 양방향(bidirectional)의 정보를 이용하는 것이 중요한 task에선 매우 치명적임

*sub-optimal*: 그냥 차선책이라는 뜻인지 아니면 다른 뜻이 있는지 궁금하다. 차선책이라 쓰였다면, sentence-level에선 bidirectional이 제일 optimal하고, unidirectional은 sub-optimal하다는 의미이다.
{: .notice--danger}

- 본 논문에서는 BERT를 제안하여 fine-tuning based approach를 향상시킨다
- BERT는 *Cloze task*에서 영감받은 "masked language model" (MLM) pre-training objective을 통해 앞서 언급한 unidirectionality constraint를 해결한다
- Masked language model은 임의로 input token 일부를 masking하고, objective는 context를 통해 이러한 masked token을 예측하는 것이다
- Left-to-right (unidirectional)한 language model pre-training과는 다르게, MLM ojbective는 representation이 left/right context를 결합(fuse)할 수 있게 한다
    - 이는 deep bidirectional Transformer를 pre-train할 수 있게 함
- 이러한 masked language model에 추가적으로 "next sentence prediction"을 이용하여 text-pair representation까지 jointly pre-train할 수 있게 한다
- 다음은 본 논문의 contribution이다
    - (1). bidirectional pre-training을 통한 language representation의 중요성을 증명함
        - Unidirectional language model을 제안한 GPT-1과 달리, BERT는 masked language model을 통해 pre-trained deep bidirectional representation을 가능케함.
        - 이는 또한 ELMo랑도 대조되는데, ELMo는 *독립적으로 학습한 left-to-right/right-to-left의 얕은 concatenation*을 사용하였음.
    - (2). pre-trained representation이 heavily-engineered task-specific archtecture의 필요성을 감소시켰음
        - BERT는 최초로 대량의 sentence-level/token-level task에 대해 SOTA를 달성한 fine-tuning approach임
        - 다양한 task-specific architecture를 압도함
    - (3). BERT는 11개의 NLP task에 대해 SOTA를 달성함

*cloze task(test)*: 지문에 빈 칸을 뚫어놓고 어떤 단어가 들어가는지 맞추는 테스트이다. 본래 사람들의 언어능력을 평가하는 테스트이지만, NLP에서도 쓰인다.

*독립적으로 학습한 left-to-right/right-to-left의 얕은 concatenation*: 살짝 헷갈렸는데, ELMo의 경우엔 forward/backward LSTM parameter $\theta$를 공유하지 않는다. 즉, 따로따로 학습한 이후에 얘네 둘을 concat한다. 원문은 다음과 같다. "This is also in contrast to Peters et al. (2018a), which uses a shallow concatenation of independently trained left-to-right and right-to-left LMs."
{: .notice--info}

# 2. Related work

패스

# 3. BERT

- 두 가지의 step이 존재
    - pre-trained
        - 다른 pre-training task에 대해 unlabeled data를 사용하여 학습
    - fine-tuning
        - pre-trained parameter로 initialize한 후, downstream task 수행하여 parameter를 fine-tuning
- 밑의 figure 1은 본 모델에서의 Q.A. 예시임

![figure 1](https://user-images.githubusercontent.com/47516855/98545193-77048f80-22d8-11eb-8ed7-c240cc273c51.png){: .align-center}

- BERT의 독특한 특징은 다른 task에 대해서도 같은 구조를 유지하고 있다는 것

**Model Architecture**

- BERT의 구조는 Transformer의 encoder
- 본 논문에서의 notation은 다음과 같음:
    - $L$: # layers
    - $H$: hidden size
    - $A$: # heads
    - 이에 더해 Feed-forward는 4H를 사용
- BERT_BASE (BASELINE) 의 스펙은
    - $L$: 12
    - $H$: 768
    - $A$: 12
- BERT_LARGE의 스펙은
    - $L$: 24
    - $H$: 1024
    - $A$: 16
- BERT_BASE의 경우 OpenAI GPT와의 비교를 위해 같은 사이즈로 만듬

**Input/Output Representation**

- BERT로 하여금 다양한 downstream task를 다루게 하기 위해, input representation으로 하여금 하나의 token sequence 안에 한 문장/문장 쌍 (e.g., <Qustion, Answer>) 모두를 표현할 수 있게 한다.
    - 본 논문에서 "sentence"이란 실제 문장이라기보단, 연속된 text의 모음이다.
    - "sequence"는 BERT에 넣을 input token sequence를 의미한다.
        - 앞서 밝힌 바와 같이, 하나의 문장이 될수도 있고, 두 개의 문장 쌍이 될 수도 있다.
- 본 논문에서는 30000개의 vocab을 갖는 WordPiece를 사용
    - 모든 sequence의 첫 번째 token은 항상 special classification token인 [CLS]가 된다.
    - 이 토큰에 해당하는 마지막 hidden state는 classification task를 위한 representation이 된다.
    - 문장 쌍의 경우에는 하나의 sequence로 합쳐서 들어가게 된다
- 본 논문은 문장들을 구분 짓는 방법이 2개 있음
    - special token [SEP]를 통해 분리하는 방법
    - 모든 토큰에 문장 A에 속하는지, B에 속하는지를 학습한 embedding을 더하는 방법
- Figure 1에서 볼 수 있듯, input embedding을 $E$, special token [CLS]의 마지막 hidden state를 $ C \in \mathbb R^H$로, i번째 인풋 토큰을 $ T _i \in \mathbb R^H$로 표현
- 토큰의 표현은 token, segment, position embedding의 합으로 구성
    - 이는 다음 그림인 figure 2에 잘 나와있음

![Figure 2](https://user-images.githubusercontent.com/47516855/98550274-3fe5ac80-22df-11eb-9b2e-a18b49868953.png)

## 3.1. Pre-training BERT