---
title:  "ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS review"
toc: true
toc_sticky: true
permalink: /project/nlp/ALBERT-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2022-05-09
---

## 들어가며

ALBERT는 parameter sharing을 통해 parameter의 양을 줄이면서 성능은 향상시킨 논문이다. 이를 위하여 embedding layer를 분해하는 factorization하고, 모든 레이어의 파라미터를 공유하는 구조를 취하고 있다. 더 작은 파라미터와 더 나은 성능을 보이지만, 그럼에도 불구하고 더 큰 구조로 인해 학습속도가 낮다는 단점이 있다.  

자세한 내용은 [논문](https://arxiv.org/abs/1909.11942)과 [깃허브](https://github.com/google-research/ALBERT)를 참고하자.

## Introduction

NLP에서 pre-training/fine-tuning scheme은 사실상의 표준이 되며 language representation learning이 갖는 한계점의 돌파구가 되었다.
이에는 워낙 많은 논문들이 있고, LM을 다루다보면 한번쯤은 보게 되는 Dai & Le, 2015; Radford et al., 2018; Devlin et al., 2019; Howard & Ruder, 2018 등이 있다.

활용 가능한 데이터가 작은 task를 포함, 다양한 non-trivial NLP tasks는 이러한 pre-training으로 부터 많은 이점을 얻었다.
이러한 연구들 중에는 대용량의 LM이 SOTA를 달성하는데 매우 중요한 역할을 하였으며, 더 큰 모델을 만들고 이로부터 distillation하는 것이 일반적인 방법이 되었다 (Sun et al., 2019; Turc et al., 2019).
이렇듯 모델의 사이즈가 중요해지는 시점에서 논문의 저자들은 다음과 같은 질문을 던진다.

**과연 더 큰 모델을 사용하는 것 만큼 더 좋은 NLP 모델을 얻는 것이 쉬울 것인가?**

이러한 질문에 대한 답변으로 가장 큰 장애물은 가용가능한 **하드웨어의 메모리**이다.
현존하는 SOTA 모델은 종종 수백만, 혹은 수십억의 파라미터를 필요로하며, 모델의 크기를 늘리려하면 이러한 메모리 문제에 맞닥뜨리게 된다.
학습속도 또한 communication overhead가 모델의 파라미터와 비례관계에 있으므로 분산 학습에서 상당한 문제가 된다.

앞서 언급한 문제에 대해 현존하는 해결법으로는 Mesh-tensorflow (Shazeer et al., 2018), Megatron-LM (Shoeybi et al., 2019)과 같은 model parallelization과, gradient checkpointing (Chen et al., 2016), RevNet(Gomez et al., 2017)과 같은 memory management이 있다.
그러나 이로는 메모리 문제만 해결할 수 있을뿐 앞서 언급한 **communication overhead는 해결하지 못한다**.

본 논문에서는 앞서 언급한 두 개의 문제를 해결하기 위해 기존의 BERT 파라미터보다 훨씬 더 소량의 파라미터를 사용하는 **A Lite BERT (ALBERT)**를 설계하였다.

ALBERT는 두 개의 parameter reduction techniques을 통해 pre-trained model의 파라미터를 늘리는데 방해가 되는 문제들을 해결한다.

1. **factorized embedding parameterization**: 큰 사이즈의 vocabulary embedding matrix를 두 개의 작은 행렬로 분해하여 hidden layer의 크기를 vocabulary embedding의 크기와 연동되지 않게한다. 이를 통해 hidden size를 키워도 embedding matrix의 size는 크게 변화하지 않는다.
2. **cross-layer parameter sharing**: 이는 네트워크의 깊이가 깊어지더라도 파라미터에는 영향이 가지 않게한다.

이 두개의 방법 모두 BERT의 성능은 해치지 않으면서 파라미터의 수는 줄일 수 있다. 따라서 parameter-efficiency를 향상시킨다.
BERT-large와 비슷한 ALBERT의 구성은 18배 작은 파라미터를 갖고 있으며, 학습 속도는 약 1.7배 빠르다.
Parameter reduction techniques은 또한 일종의 정규화로 동작하며, 학습의 안정성과 일반화를 돕는다.

또한, ALBERT의 성능을 더욱 향상시키기 위해, sentence-order prediction (SOP)라 불리는 loss를 새로 도입한다.
SOP는 문장 간 응집성에 초점을 맞췄으며, XLNet, RoBERTa에서 지적했던 NSP의 비효율성을 해결한다.

이러한 디자인을 통해 훨씬 큰 ALBERT를 만들 수 있었으며, 이럼에도 불구하고 BERT-large보단 더 작은 파라미터를 갖으며 성능은 훨씬 더 좋은 결과를 보였다.


## RELATED WORK

사전연구는 보통 작성하진 않지만, 이번에는 모르는 내용이 조금 있어 정리해볼까 한다.

### SCALING UP REPRESENTATION LEARNING FOR NATURAL LANGUAGE

본 섹션은 LM을 공부한 사람이라면 대부분은 알만한 내용이다.

Pre-training/fine-tuning 구조에서, 대용량 모델은 성능을 향상시키는 경우가 많았다. 
이의 대표적인 예가 BERT이다.
BERT의 경우는 **큰 hidden size와 layer, attention head**를 사용하는 것이 항상 **성능향상**을 이끔을 보였다.
그러나 이들도 hidden size 1024에서 실험을 멈추었는데, 예측컨데 모델 사이즈와 **computation cost**로 인한 문제로 보인다.

연산 능력의 제한으로 큰 모델로 실험하는 것은 어렵다.
현존하는 SOTA가 수백만/수십억의 파라미터를 갖는 점을 고려하였을 때, 메모리 이슈가 일어날 것이란게 자명하다.

이를 해결하기 위해 [Chen et al. (2016)](https://arxiv.org/pdf/1604.06174.pdf)는 *gradient checkpointing*를 제안하여 추가적인 forward pass에서의 메모리 요구사항을 sublinear하게 감소시켰다.

[Gomez et al. (2017)](https://arxiv.org/abs/1707.04585)은 RevNet을 제안하여 다음 레이어에서 각 레이어의 activation을 재구축하는 방식을 통해 intermediate activations을 저장하지 않는 방법을 사용했다.

이러한 두 방법 모두 메모리 용량을 줄여 속도 향상을 이끌어내었다.

T5는 대용량의 모델을 학습할 때 model parallelization을 사용하였다. 반면, ALBERT는 parameter-reduction techniques을 사용하여 메모리 사용량을 줄이고 학습속도를 높혔다.

### CROSS-LAYER PARAMETER SHARING

파라미터를 공유하는 연구는 이전에도 존재하였지만, 일반적인 Transformer구조에만 한정되고 pre-training/fine-tuning scheme에서는 연구된 바가 없다.
본 논문에서의 발견과 다르게 Universal Transformer에서는 레이어 간 파라미터 공유가 LM과 주어-동사 일치 문제에서 더 나은 성능을 보임을 발견하였다.

DEQ의 경우 Transformer를 이용, 특정 레이어에서 input/output embedding이 동일해지는 평형점에 도달할 수 있음을 보였다.


### SENTENCE ORDERING OBJECTIVES

ALBERT는 두개의 연속된 segment의 순서를 맞추는 loss를 갖는다.
이러한 담화 응집성(discourse coherence)과 관련된 연구도 여럿 존재한다.

담화에서의 응집성과 결속성 널리 연구되어 왔으며, 이웃하는 segment를 연결해주는 현상 또한 연구되어 왔다.

> 응집성 (coherence): 텍스트에 포함되어 있는 내용들 간의 **의미적인 연결 관계**  
> 결속성(cohesion): 텍스트에 포함되어 있는 요소(문장)들을 연결해 주는 **표면적인 언어 자질**. 

실제로 효과가 있다고 밝혀진 대부분의 objective는 대게 단순하다.
Skip-thought/FastSent sentence embedding은 **이웃하는 문장의 단어를 예측**하는 방식으로 문장을 학습한다.
또 다른 연구로는 바로 **이웃하는 문장보다 더 나중 문장을 예측**하거나 담화표지(discourse marker: 대화에서 특정한 역할을 해주는 것)를 예측하는 것이 있다.

> 담화표지(discourse marker): 주로 구어에서, 문장의 내용에 직접적인 영향을 미치지는 않지만 전체적인 분위기나 대화의 최종적인 목적을 달성하고자 문장 간의 응집성을 높이기 위하여 사용하는 표지. 화자의 상태나 의도, 감정을 나타내기도 한다.

본 연구는 두 개의 연속하는 문장의 순서를 예측하는 방식으로 sentence embedding을 학습하는 Jernite et al. (2017)의 연구와 비슷하다.
그러나 이러한 연구들과 ALBERT의 SOP의 가장 큰 차이는 문장 단위가 아니라 text segment 단위로 동작한다는 것이다.

## THE ELEMENTS OF ALBERT

여기서는 ALBERT의 design choice와 BERT와의 정량적 비교를 진행해보도록 하겠다.

### MODEL ARCHITECTURE CHOICES

ALBERT의 backbone은 BERT와 유사하게 Transformer 인코더와 GELU 활성화함수를 쓴다는 면에서 비슷하다.
BERT의 notation을 따라 vocabulary embedding size를 $E$로, 인코더 레이어의 갯수를 $L$로, hidden size를 $H$로 표현한다.
BERT와 마찬가지로 feed-forward/filter size를 $4H$로, attention head를 $H/64$로 설정한다.

BERT의 디자인에서 향상을 일구어낸 ALBERT의 주요 contribution은 다음과 같다.

**Factorized embedding parameterization**

BERT와 이의 후속작 XLNet, RoBERTa를 포함하여 WordPiece embedding size $E$는 hidden layer size $H$에 연동되어 있다.
즉, $E \equiv H$이다.
이는 모델링과 실용적인 이유에서 suboptimal하다.

먼저 모델링관점에서 살펴보자. 

WordPiece embedding은 어떠한 입력이 들어오더라도 같은 값을 내놓으므로 **context-independent** representation을 학습하도록 되어있다. 반면 hidden-layer embeddings은 단어의 맥락에 따라 달라지므로 **context-dependent** representations을 학습하도록 되어있다.
[RoBERTa](/project/nlp/roberta-review/)는 context length가 성능에 미치는 영향을 실험하였는데, 이처럼 BERT-like representation의 성능은 context를 이용하여 context-dependent representations을 학습하는데서 온다.
이와같이 WordPiece embedding size $E$와 hidden layer size $H$를 분리하는 것은 모델링 요구사항에 따라 (즉, $H \gg E$) 전체적인 모델 파라미터를 더욱 효율적으로 사용할 수 있게끔 한다.

실용적인 관점에서 살펴보면, NLP는 일반적으로 vocab size $V$가 커야한다. 만일 $E \equiv H$라면, $H$가 커질수록 embedding matrix $E \times V$가 커지게된다.
이로 인해 모델은 수십억개의 파라미터를 갖게되고, 대부분은 학습동안 sparse하게 업데이트 될 것이다.

따라서 ALBERT는 **embedding parameter를 두 개의 작은 matrix로 분해**한다.
One-hot vector를 직접 크기 $H$의 hidden space로 projection하는 대신, 크기 $E$의 lower dimensional embedding space로 먼저 projection하고, 이후 hidden space로 한번 더 projection한다.
이러한 분해를 통해 embedding parameters를 $O(V \times H)$에서 $O(V \times E + E \times H)$로 줄일 수 있게된다.
이는 특히 $H \gg E$일 때 더욱 중요하다.

ALBERT는 모든 word pieces에 대해 같은 $E$를 사용한다. 
이는 전체 단어의 embedding에 비해 훨씬 문서들 사이에 더 균등하게 분포되어있고, 다른 단어에 대해 다른 embedding size를 갖는 것이 중요하기 때문이다 (Grave et al. (2017); Baevski & Auli (2018); Dai et al. (2019)).

**각 단어가 같은 사이즈의 임베딩**을 갖는건 아주 당연한 상식같다. 
하지만 사전연구를 얼추 살펴보았을 때 **단어 빈도수에 따라 임베딩 사이즈를 늘리는** 등 **각 단어마다 다른 사이즈의 임베딩**을 이용하는 연구들도 있기 때문에 본 연구에서는 이를 사용하지 않는다는 뜻으로 보인다.
다만 Transformer-XL은 이와 관련이 없는 것 같은데 무슨 이유로 들어갔는지 잘 모르겠다.

![image](https://user-images.githubusercontent.com/47516855/168437979-36b97d01-9a39-407d-9f7e-e6737b51bcd9.png){: .align-center}{: width="600"}

**[Adaptive input representations (Baevski & Auli, 2018)](https://arxiv.org/pdf/1809.10853.pdf)의 architecture**. Embedding size가 단어의 빈도수에 따라 달라진다.
{: .text-center}


**Cross-layer parameter sharing**

ALBERT에선 파라미터 효율을 위한 방법으로 **레이어 간에 파라미터 공유**를 제안한다.
파라미터를 공유하는 방법에는 여러가지 방법이 있는데, 이는 레이어 간 feed-forward network (FFN)만 공유하거나, attention paramter만 공유하는 방법이다.
여기서는 **레이어 간 모든 파라미터**를 공유한다.
따로 명시하지 않는 이상 이러한 방법을 디폴트로 사용한다.

이와 비슷한 전략으로는 Universal Transformer(UT)과 Deep Equilibrium Models(DEQ)가 있다.
ALBERT의 결과와는 달리 UT는 vanilla Transformer의 성능을 압도하였다.

DEQ는 implict layer의 일종으로, **deep neural network를 단 하나의 레이어**를 통해 표현하는 모델이다.
즉, 모든 레이어는 동일한 파라미터를 갖으며, 무한히 깊은 레이어를 쌓았을 때, 레이어의 인풋이 아웃풋과 동일해지는 지점이 있다는 것이다.
이를 평형점(equilibrium point)이라하며, 일반적인 네트워크를 학습시키는 대신 이 평형점을 직접 찾아내는 것을 목표로 한다.

![image](https://user-images.githubusercontent.com/47516855/168438937-ef12201e-eff5-4403-8f5f-4a33c6515d1f.png){: .align-center}{: width="600"}

DEQ는 자연어처리 등 다양한 모델에서 실험하여 파라미터의 양은 훨씬 적으면서 거의 비슷한 성능을 내었다. 

ALBERT에서는 DEQ에서 **input/output embedding이 동일한**것과 다르게, L2 distances와 cosine similarity가 0으로 수렴하는 대신 진동하는 것을 찾아내었다.

![image](https://user-images.githubusercontent.com/47516855/168418714-881fb7f3-7332-4a17-a3e2-3c24a881109b.png){: .align-center}{: width="700"}

위 그림은 각 레이어의 input/output embedding에 대한 L2 distances와 cosine similarity 값을 나타낸 것이다.
여기서 두 레이어의 값을 비교하기 위해 L2 distance를 사용한 것을 볼 수 있다.
만일 두 레이어의 값이 같다면 **L2 distance는 0으로 수렴**해야 할 것이다.
그러나 ALBERT는 수렴하는 대신 특정 값으로 진동하는 것을 확인하였다.

또한, 레이어를 이동할 때 마다 ALBERT가 BERT보다 훨씬 더 부드러운 것을 볼 수 있는데, 이를 통해 weight-sharing이 **파라미터의 안정화**에 영향을 미치는 것을 알 수 있다 (특정 값으로 수렴).
비록 BERT와 비교했을 때 초반에 급격하게 하락하는 모습을 보였으나, 그럼에도 불구하고 24레이어가 지날 때 까지 0에 수렴하지 않는 것을 확인할 수 있다.
이는 **ALBERT 파라미터의 solution space가 DQE의 solution space와는 매우 다르다**는 것을 보여준다.

**Inter-sentence coherence loss**

NSP는 문장 쌍의 관계를 추론하는 NLI와 같은 downstream task에서 사용하기 위해 도입되었다.
그러나 XLNet과 RoBERTa에 의하면 NSP는 성능에 영향을 끼치지 못한다.
NSP의 효과가 없는 이유는 MLM과는 달리 **task가 어렵지 않기 때문**으로 짐작된다.

NSP는 *토픽 예측(topic prediction)*과 *응집성 예측(coherence prediction)*을 하나의 task로 결합한 형태로 볼 수 있다. 
이는 NSP를 수행할 때 negative-example의 경우 두 segment를 다른 문서에서 가져오기 때문에 토픽/응집성 측면에서 어울리지 않기 때문이다 (misaligned).
그러나 토픽 예측은 응집성 예측에 비해 학습하기 쉽고 MLM loss와 겹치는 부분이 있다.

본 논문의 저자들은 문장 간 모델링(inter-sentence modeling)이 language understanding의 중요한 측면 중 하나로 주장하지만, **응집성**에 더욱 기반을 둔 **Sentence-order prediction (SOP) loss**를 제안한다.
SOP loss는 토픽 예측을 제거하는 대신 문장 간 응집성(inter-sentence coherence)을 모델링한다.

SOP loss는 positive examples (같은 문서에서 추출한 연속된 segment)는 BERT와 똑같은 테크닉을 사용하고, **negative examples은 똑같이 연속된 segment지만 순서가 뒤바뀐 것**을 사용한다.
이를 통해 모델이 문서수준에서의 응집성 속성에 대해 더욱 세분화된 차이점(finer-grained distinctions)을 학습하도록 한다.
추후 살펴보겠지만 이는 NSP는 SOP task를 풀 수 없으며 (즉, 더 쉬운 토픽 예측의 신호를 학습하며, SOP task에 대한 임의의 baseline 수준만 수행), 반면 SOP는 예측컨데 불일치된 응집성의 단서 (misaligned coherence cue)를 분석하는 식으로 NSP task를 합리적인 수준에서 풀 수 있었다.
이 결과 ALBERT는 일관적으로 multi-sentence encoding tasks에 대한 downstream task에 대한 성능 향상시킴을 확인하였다.

### MODEL SETUP

여기서는 BERT와 ALBERT의 hyperparameter를 비교해보도록 한다.
앞서 언급한 design choice로 인해 ALBERT는 BERT에 비해 훨씬 더 작은 파라미터를 갖는다.

![image](https://user-images.githubusercontent.com/47516855/168420421-f566198e-e8d6-4076-a7f4-16c4c6a09bec.png){: .align-center}{: width="600"}

ALBERT-large의 경우 BERT-large에 비해 18배 적은 파라미터를 갖는다 (18M vs. 334M).
ALBERT-xlarge의 구성은 $H=2048$로, 오직 60M의 파라미터를 갖는다.
ALBERT-xxlarge의 구성은 $H=4096$로, 오직 233M의 파라미터를 갖으며, 이는 BERT-large의 70%에 해당한다.
ALBERT-xxlarge의 경우 24레이어와 12레이어 모델의 성능에 차이가 없으므로 연산이 더 적은 12레이어 짜리를 사용한다.

파라미터 효율성에 대한 향상은 ALBERT의 가장 중요한 이점이며, 이러한 이점을 정량화하기에 앞서 실험 과정을 더 살펴보도록 한다.

## EXPERIMENTAL RESULTS

### EXPERIMENTAL SETUP

가능한 의미있는 비교를 위해 BERT와 동일한 configuration에다가 BookCorpus, Wikipedia 데이터를 통해 학습하여 baseline을 만든다 (총 16GB 텍스트).

입력 포맷은 [CLS] $x _1$ [SEP] $x _2$ [SEP]의 형태가 된다.
여기서 $x _i$는 segment가 된다.
최대 입력 길이는 512로 제한하며, 10%의 확률로 512보다 작은 임의의 문장들을 생성한다.
BERT처럼 사전의 크기는 30,000을 사용하며, SentencePiece를 사용하여 만들었다.

MLM에서 사용할 마스킹된 단어는 SpanBERT의 $n$-gram masking (span masking)을 사용한다.
마스킹할 단어의 길이 $n$은 다음과 같은 확률로 구해진다.

$$
p(n) = \frac{1/n}{\sum^N _{k=1} 1/k}
$$

마스킹할 단어의 최대 길이는 3으로 설정한다.

Batch size는 4096을 사용하며, 배치 사이즈가 크기 때문에 안정적으로 학습할 수 있는 LAMB optimizer (lr=0.00176)를 사용한다.
LAMB optimizer에 대한 설명은 [다음](https://junseong.oopy.io/paper-review/lamb-optimizer)을 참고해보자.
모든 모델은 특별한 언급이 없는 한 125,000 스텝으로 학습했다고 보면된다.

### OVERALL COMPARISON BETWEEN BERT AND ALBERT

이제 ALBERT design choice에 대한 영향력을 측정해보자. 
파라미터 효율성에 대한 결과는 다음 테이블에 나와있다.

![image](https://user-images.githubusercontent.com/47516855/169252012-068e04a5-7cf4-4231-90e3-1f7f6601f607.png){: .align-center}{: width="600"}

BERT-large의 70%의 파라미터만으로 ALBERT-xxlarge는 BERT-large에 비해 굉장한 성능 향상을 보였다.
Dev set에 대한 실험결과는 SQuAD v1.1 (+1.9%), SQuAD v2.0 (+3.1%), MNLI (+1.4%), SST-2 (+2.2%), RACE (+8.4%)로 나타났다.

또한 같은 조건하에서 학습 시 데이터 쓰루풋(throughput)의 속도 차이도 굉장히 흥미로운데, 더 적은 communication과 연산으로도 더 높은 쓰루풋을 보였기 때문이다. 
여기서 쓰루풋은 **특정 단위 시간 당 특정 데이터 양의 처리량**을 의미한다.
BERT large를 baseline로 봤을 때 (1.0), ALBERT-large는 1.7배 빠른 속도를 보였고, ALBERT-xxlarge의 경우 더 큰 구조 때문에 약 3배 정도 느린 모습을 보여줬다.
그러나 **파라미터의 양이 더 적음에도 불구하고** 학습 속도에 차이가 나는 이유는 논문에서 밝히지 않고 있다.

### FACTORIZED EMBEDDING PARAMETERIZATION

아래 테이블은 ALBERT base의 세팅에서 embedding size $E$의 변화에 따른 효과를 파악한 것이다.

![image](https://user-images.githubusercontent.com/47516855/169636239-fc4115bb-813e-484c-8085-53739622ea4c.png){: .align-center}{: width="700"}

BERT와 같이 파라미터를 공유하지 않는 경우 embedding size가 클 경우 더 좋은 성능을 보였으나 그 차이는 미미했다.

ALBERT와 같이 파라미터를 공유하는 경우 embedding size 128으로 사용하는 것이 제일 좋은 성능을 보였다.
이러한 결과에 기반하여 앞으로는 $E=128$로 세팅한다.

### CROSS-LAYER PARAMETER SHARING

아래 테이블은 다양한 레이어 간의 파라미터 공유에 대한 실험 결과이다.

세팅은 ALBERT base 환경에 두 개의 embedding size를 사용한다 ($E=768, E=128$).
마찬가지로 파라미터 공유/비공유 환경에서 비교를 진행하였고, 오직 attention parameter만 공유/FFN만 공유하는 실험을 추가하였다.

![image](https://user-images.githubusercontent.com/47516855/169637637-074a22dc-257e-43ec-9e94-f801dc6e97f0.png){: .align-center}{: width="800"}

모든 파라미터를 공유하는 경우 embedding size에 관계없이 성능이 하락하였으나 $E=128$일 때의 성능폭이 작았다.

추가로 FFN-layer를 공유하는 경우 대부분 성능하락이 발생하였다.
반면 attention layer만을 공유하는 경우 $E=128$일 때는 성능 하락이 발생하지 않았으며, $E=768$일 때는 약간의 하락이 발생하였다.

이외에도 레이어 간 파라미터를 공유하는 방법이 있는데, $L$개의 레이어를 size $M$의 $N$ 그룹으로 나누어 각 size $M$의 그룹이 파라미터를 공유하는 것이다.
실험결과 $M$이 작을수록 더 좋은 성능을 내었으나 파라미터도 급격하게 커지게 된다.
ALBERT에서는 모든 파라미터 공유 (표의 all-shared)하는 것을 디폴트 세팅으로 잡았다.

### SENTENCE ORDER PREDICTION (SOP)

여기서는 none (XLNet, RoBERTa), NSP (BERT), SOP (ALBERT) 세 개의 문장 간 모델링에 대한 loss에 대해 비교실험(head-to-head)을 진행하였다.

표의 intrinsic의 경우 SQuAD, RACE의 dev set에 대해 정확도를 측정한 것이며,
이 경우 모델의 loss가 수렴하는지 확인하는 용도로 사용한 것이지 model selection 용도로 사용하진 않았다.

Downstream evaluation의 경우 dev set에 대해 early stopping을 사용하였다.

![image](https://user-images.githubusercontent.com/47516855/169638146-fafc93fd-578c-4479-aeee-d9a5005bfbd4.png){: .align-center}{: width="700"}

Intrinsic tasks의 경우 NSP는 SOP에 대해 별 다른 효과를 보여주지 못하였다 (52.0%의 정확도로 아무것도 하지 않는 none과 비슷).
이를 통해 NSP가 토픽의 변화에 대해서만 모델링한다고 결론낼 수 있었다.

반면 SOP는 NSP에서 78.9%라는 상대적으로 높은 성능을 보였고, SOP에서는 86.5%를 달성하였다.
이보다 더욱 중요한 것은 SOP의 경우 **multi-sentence encoding task에서 일관되게 성능 향상**을 보였다는 것이다.

### WHAT IF WE TRAIN FOR THE SAME AMOUNT OF TIME?

이번 파트는 상당히 흥미로운 부분인데, 저자들은 이에 추가로 **같은 시간동안 학습한 결과**를 비교하였다.

앞선 Table 2에서 보여준 속도를 보면 BERT-large에 대한 데이터 쓰루풋이 ALBERT xx-large에 비해 3.17배 높은 것을 볼 수 있다.
일반적으로 더 많이 학습시킬 수록 더 좋은 성능을 내기 때문에 여기서는 단순 데이터 쓰루풋 (즉, 학습 step 수)가 아닌 실제 학습 시간으로 성능을 비교하였다.

BERT-large의 경우 400k step (34시간), ALBERT-xxlarge의 경우 125k step (32시간)를 학습시켰다.

![image](https://user-images.githubusercontent.com/47516855/169639006-3edfeca8-3271-43b4-a84c-d2f3243a04f5.png){: .align-center}{: height="300"}

ALBERT-large의 경우 BERT-large보다 평균 1.5% 높으며, RACE에서는 5.2% 향상된 결과를 보였다. 

### ADDITIONAL TRAINING DATA AND DROPOUT EFFECTS

이번에는 Wikpedia와 BookCorpus에 추가로 XLNet과 RoBERTa의 데이터를 사용해보겠다.

![image](https://user-images.githubusercontent.com/47516855/169640354-52c54acf-85e3-460d-b0bc-649f69739868.png){: .align-center}{: width="600"}

아래 그림의 a는 MLM에 대한 dev set 결과로, **데이터를 추가로 사용할 경우 성능이 크게 증가**하는 것을 확인할 수 있다.

또한, 아래 표는 downstream task에서의 성능으로, SQuAD의 경우 위키피디아 기반으로 만들어져있기 때문에 out-of-domain의 문제로 제외하였다.

![image](https://user-images.githubusercontent.com/47516855/169642502-d79c9f00-255a-4792-a998-491412786533.png)

그 결과 1M step의 학습에도 불구하고 가장 큰 모델의 경우 **오버피팅하지 않는 것**을 확인하였다.
따라서 모델의 용량을 더 크게 만들기 위해 dropout을 제거한 모델 (위 그림 Fig. 2b)을 살펴보았고, MLM에서 더 나은 성능을 보여주는 것을 확인하였다.

CNN에서는 batch normalization과 dropout이 성능을 저하시킬 수 있다는 경험적/이론적 증거가 있는데, 본 논문은  large Transformer-based model에서도 dropout이 성능을 저하시킬 수 있다는 것을 밝혀낸 첫번째 논문이다.
그러나 ALBERT의 기저는 Transformer의 특정한 종류이기 때문에 이러한 현상이 다른 Transformer 구조에서도 발견되는지에 대해서는 추가적인 실험이 필요할 것이다.

### CURRENT STATE-OF-THE-ART ON NLU TASKS

이번에는 추가 데이터를 활용하여 학습한 ALBERT와 SOTA를 비교해보자.

크게 두 가지 세팅에서 비교를 하였는데, 하나는 단일 모델이고, 나머지 하나는 앙상블을 이용하였다.
둘 모두 RoBERTa를 따라서 MNLI의 checkpoint에서 RTE, STS, MRPC 데이터를 fine-tuning하였다.
앙상블 checkpoint의 경우 dev set 기준으로 6-17개 사이의 모델을 골라 만들었다.

GLUE/RACE는 앙상블의 평균치로 결과를 보고하였으며, 12/24 layer 구조를 활용하였다.
SQuAD의 경우 span에 대한 확률값의 평균을 사용하였다.

결과보고는 RoBERTa와 같이 dev set에서 다섯번 돌린 결과의 중앙값으로 진행하였다.
ALBERT의 경우 지금까지의 실험에서 최고의 환경만으로 구성하였다 (xxlarge, MLM + SOP loss, dropout 제거)

![image](https://user-images.githubusercontent.com/47516855/169643004-979302ca-b54f-48cd-80f4-a30717467b87.png){: .align-center}{: width="800"}

![image](https://user-images.githubusercontent.com/47516855/169643085-15c2e177-0596-488a-958a-c9c5a000ef3a.png){: .align-center}{: width="600"}

단일/앙상블 모델 모두 SOTA에 비해 상당한 성능향상을 이끌어내었다.
특히 RACE는 큰 폭으로 증가하였는데, 앙상블 모델이 이러한 MRC에 대해 특화된 구조이기 때문이다.
단일 모델의 경우 SOTA 앙상블 모델에 비해서도 2.4% 상승한 모습을 볼 수 있다.
