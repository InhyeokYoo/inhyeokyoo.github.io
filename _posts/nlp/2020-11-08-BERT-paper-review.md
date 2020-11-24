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
last_modified_at: 2020-11-12
---

이번 시간엔 NLP pre-trained model의 서막을 알린 BERT에 대해 알아보자. 원문은 [다음 링크](https://arxiv.org/abs/1810.04805)에서 확인할 수 있다.

우선 논문을 읽고 궁금증과 이에 대한 설명/해석을 적었다. 다음은 논문을 읽으면서 생긴 궁금증이다.

- 왜 BERT가 대단한가?
    - bidirectional이 그렇게 대단한가?
        - unidirectional은 안 좋은가?
        - 이전 논문들은 왜 시도를 안(못) 했지?
- 왜 GPT는 decoder를 썻지?
- encoder-decoder 둘 다는 안 되나?
- GPT와의 성능 차이: encoder vs. MLM vs. NSP ?


# 1. Introduction

Language modeling pre-training은 다양한 NLP task에서 효과적임을 보여왔다. 이에 대표적인 예로는,
- GPT-1
- ELMo
- *Semi-supervised sequence learning*<sup>1</sup>
- *ULMFiT*<sup>2</sup>
등이 있다. 이는 language inference, *paraphrasing*<sup>3</sup>과 같이 문장들을 일부분이 아닌 전체적으로 분석하여 이들 사이의 관계를 예측하는 **sentence-level task**와, name entity recognition (NER), question answering(QA)와 같이 모델이 token 단위로 fine-grained output을 내는 **token-level task**을 포함한다.

*1. Semi-supervised sequence learning*: 이 논문은 2015년도에 나온 논문인데, unlabeld data를 이용하여 sequence learning을 진행한 논문이다. 일반적인 LM ($p(x _t \rvert x _1, ..., x _{t-1})$)과 sequence autoencoder를 써서 괜찮은 결과를 뽑아냈다는 논문이다. 지금 생각해보면 그냥 그렇지만, 2015년에 나온걸 감안하면 세대를 앞서갔다는 느낌도 든다. 원문은 [다음](https://arxiv.org/pdf/1511.01432.pdf)을 참고.
{: .notice--info}

*2. ULMFiT*: 앞선 GPT-1에서 보았던 discriminative fine-tuning을 제안한 논문이다. 이 논문은 general-domain corpus에 대해 pre-train하고, target task에 대한 data에 대해 discriminative fine-tuning과 slanted triangular learning rates를 이용하여 fine-tuning을 진행한 뒤, classifier를 gradual unfreezing, discriminative fine-tuning, slanted triangular learning rates를 이용하며 fine-tuning한다고 한다.
{: .notice--info}

*3. paraphrasing*: paraphrase를 생성하거나 탐지하는 작업을 말한다. 위키피디아에서는 QA도 paraphrasing이라고 설명하고 있는데도 불구하고 BERT에서 QA가 token-level task인 이유는 각 token-level에 대해서 정답의 시작점과 끝점의 index를 찾기 때문이다.
{. :notice--info}

pre-trained language representation을 downstream task에 적용하는 방법은 두 가지가 있다고 한다.

**1. feature-based**

pre-trained representation을 **추가적인 feature**로 이용하는 **task-specific architecture**를 사용하는 방법이다. ELMo가 그 대표적인 예시이다. 다음은 ELMo 논문의 3.3장에서 발췌한 내용이다.

> Given a pre-trained biLM and a supervised architecture for a target NLP task, it is a simple processto use the biLM to improve the task model.  Wesimply  run  the  biLM  and  record  all  of  the  layerrepresentations  for  each  word.   Then,  we  **let the end task model learn a linear combination of these representations**, as described below.

**2. fine-tuning approach**

최소한의 task-specific parameter만을 도입하는 모델이다. downstream에 대해 단순히 모든 parameter를 fine-tuning한다. 이에는 GPT-1이 대표적이다. ([리뷰 보러가기](/project/nlp/gpt1/)) 

이 두 가지 방법은 pre-training에서 같은 objective function ($p(x _t \rvert x _1, ..., x _{t-1})$)을 공유하고, unidirectional language model을 통해 일반적인 language representation을 학습한다.

그러나 논문에서는 이러한 방법이 pre-trained representation의 능력을, 특히 fine-tuning 방법에서, **제약**시킨다고 보고 있다 (We argue).
- 가장 큰 제약은 일반적인 LM이 **단방향**이라는 것이고, 이로 인해 pre-training에서 사용하는 **모델의 선택지를 제한**한다고 본다.
    - 예를 들어 GPT의 경우엔 left-to-right architecture (transformer decoder)를 사용하였는데, 이는 subsquence mask로 인해 오직 **이전의 token**에만 self-attnetion이 가능하다.
- 이러한 제약은 sentence-level에선 *sub-optimal*이고, token-level task (e.g. QA)와 같이 양방향 (bidirectional)의 정보를 이용하는 것이 중요한 task에선 매우 치명적이다.

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
- 밑의 Figure 1은 본 모델에서의 Q.A. 예시임

![Figure 1](https://user-images.githubusercontent.com/47516855/98634518-c0c69780-2323-11eb-820b-20663a1ac8fa.png)
{: .align-center}

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
    - 이는 다음 그림인 Figure 2에 잘 나와있음

![Figure 2](https://user-images.githubusercontent.com/47516855/98550274-3fe5ac80-22df-11eb-9b2e-a18b49868953.png)

## 3.1. Pre-training BERT

- ELMo나 GPT-1과는 다르게 left-to-right나 right-to-left LM같은 전통적인 모델을 사용하진 않음.
- 대신 두 개의 unsupervised task를 이용하여 pre-train함
- 이번 step은 앞선 Figure 1의 왼쪽 그림에 속함

**Task #1: Masked LM**

- 직관적으로 생각해보면 deep bidirectional model이 left-to-right model (GPT-1)이나 left-to-right/right-to-left의 shallow concatenation (ELMo)보다는 더욱 강력할 것이고, 이는 매우 그럴듯한 생각이다
- 불행하게도, *standard conditional LM은 오직 left-to-right이나 right-to-left로만 학습할 수 있다*
    - 왜냐면 bidirectional conditioning은 각 단어들로 하여금 **간접적으로** 각 단어를 볼 수 있게하고,
    - *model은 multi-layer에서 target word를 명확하게 예측*하기 때문임
- Deep bidirectional representation을 학습하기 위해서는, input 토큰에 일부에 임의로 mask를 씌운 후, 이 토큰이 무엇인지를 예측함
    - 이를 "masked LM" (MLM) 이라고 부름
    - 다른 문헌에서는 종종 Cloze task라고도 부름
- 이 경우 masked token에 해당하는 마지막 hidden vector가 입력되어 vocabulary에 대해 softmax를 계산하게 됨
    - 실험에서는 각 sequence의 15%가 (WordPiece token) 랜덤으로 mask
    - DAE가 전체 input을 복원하는 반면, 본 논문은 단순히 masked token을 예측하기만 함
- 그러나 이런 pre-train과 fine-tuning 사이에는 mismatch가 존재함
    - 왜냐면 [MASK] 토큰은 fine-tuning 단계에서 등장하지 않기 때문
    - 이를 보완하기 위해 항상 mask된 단어를 [MASK] 토큰으로 변경하는 대신,
        - (1). 80%의 확률로 [MASK]로 변경하고,
        - (2). 10%의 확률로 임의의 토큰으로 변경,
        - (3). 10%의 확률로 그대로 냅둠
    - 그 후 $T _i$는 cross entropy loss를 통해 원래의 token을 예측하는데 사용됨
- 이에 대한 다양한 실험 결과는 Appendix C.2 참고

본문에서 *standard conditional LM은 오직 left-to-right이나 right-to-left로만 학습할 수 있다*고 밝히고 있는데, 왜 그런지 이해가 안 된다. ELMo의 경우 parameter sharing을 하진 않지만 여튼 bidirectional한거 같고, 다양한 literature에도 bidirectional한 경우가 있는 것 같은데...
{: .notice--danger}

**TASK #2: Next Sentence Prediction (NSP)**

- N.L.I.나 .Q.A. 같은 downstream task는 두 문장의 **관계**에 대한 이해가 필요함
    - 그러나 안타깝게도 이는 language modeling을 이용해 직접적으로 파악할 순 없음
- 이러한 관계를 이해하기 위해 이진화 된(binarized) next sentence prediction task를 pre-train함
    - 이는 어떠한 monolingual corpus라도 이로부터 생성이 가능
- 각 pre-training example에 대해 sentence A와 B를 선택할 때,  절반의 확률로 B가 실제로 A 다음에 오는 경우이고 (*IsNext*로 label), 나머지 절반은 corpus로부터 추출한 random sentence가 된다 (*NotNext*로 label)
- Figure 1에서 $C$는 next sentence prediction으로 쓰임 (빨간색 네모)
    - 최종 모델은 NSP에서 97%-98%를 달성
- 굉장히 단순한 작업임에도 불구하고 QA와 NLI에서 매우 유용함을 증명함
    - vector $C$는 fine-tuning이 아닐 경우 유용하지 않음. 이는 NSP를 위해 학습되는 요소기 때문.
    - Section 5.1에서 확인
- NSP task는 Jernite et al. (2017)와 Logeswaran  and  Lee  (2018)의 representation-learning objective와 관련이 깊음
    - *이전 연구들은 오직 sentence embedding만 transfer하여 downstream task로 진행*
    - *반면 BERT는 end-task model paramter를 초기화하기 위해 모든 parameter를 transfer함*

무슨 말인지 명확하게 이해가 가질 않는데, 예상하기로는 이전 연구들은 embedding vector만 얻어서 downstream task를 진행하는 반면, BERT는 pre-trained model의 모든 parameter를 fine-tuning한다는 의미로 보인다.
{: .notice--info}

# 5.1. Effect of Pre-training Tasks