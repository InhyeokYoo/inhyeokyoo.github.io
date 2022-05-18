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

[논문](https://arxiv.org/abs/1909.11942)과 [깃허브](https://github.com/google-research/ALBERT)

NLP에서 pre-training/fine-tuning scheme은 사실상의 표준이 되며 language representation learning 한계점의 돌파구가 되었다.
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
parameter reduction techniques은 또한 일종의 정규화로 동작하며, 학습의 안정성과 일반화를 돕는다.

또한, ALBERT의 성능을 더욱 향상시키기 위해, sentence-order prediction (SOP)라 불리는 loss를 새로 도입한다.
SOP 주로 inter-sentence coherence에 초점을 맞췄으며, XLNet, RoBERTa에서 지적했던 NSP의 비효율성을 해결한다.

이러한 디자인을 통해 훨씬 큰 ALBERT를 만들 수 있었으며, 이럼에도 불구하고 BERT-large보단 더 작은 파라미터를 갖으며 성능은 훨씬 더 좋은 결과를 보였다.


## RELATED WORK

사전연구는 보통 작성하진 않지만, 이번에는 모르는 내용이 조금 있어 정리해볼까 한다.

### SCALING UP REPRESENTATION LEARNING FOR NATURAL LANGUAGE

본 섹션은 LM을 공부한 사람이라면 대부분은 알만한 내용이다.

Pre-training/fine-tuning 구조에서, 대용량 모델은 성능을 향상시키는 경우가 많았다. 
이에는 BERT가 대표적인 예이다.
BERT의 경우는 큰 hidden size와 layer, attention head를 사용하는 것이 항상 성능향상을 이끔을 보였다.
그러나 이들도 hidden size 1024에서 실험을 멈추었는데, 예측컨데 모델 사이즈와 computation cost로 인한 문제로 보인다.

연산 능력의 제한으로 큰 모델로 실험하는 것은 어렵다.
현존하는 SOTA가 수백만/수십억의 파라미터를 갖는 점을 고려하였을 때, 메모리 이슈가 일어날 것이란게 자명하다.

이를 해결하기 위해 [Chen et al. (2016)](https://arxiv.org/pdf/1604.06174.pdf)는 gradient checkpointing를 제안하여 추가적인 forward pass에서의 메모리 요구사항을 sublinear로 감소시켰다.

[Gomez et al. (2017)](https://arxiv.org/abs/1707.04585)은 RevNet을 제안하여 다음 레이어에서 각 레이어의 activation을 재구축하는 방식을 통해 intermediate activations을 저장하지 않는 방법을 사용했다.

이러한 두 방법 모두 메모리 용량을 줄여 속도 향상을 이끌어내었다.

T5는 대용량의 모델을 학습할 때 model parallelization을 사용하였다. 반면, ALBERT는 parameter-reduction techniques을 사용하여 메모리 사용량을 줄이고 학습속도를 높혔다.

### CROSS-LAYER PARAMETER SHARING

### SENTENCE ORDERING OBJECTIVES

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

실용적인 관점에서 살펴보면, NLP는 주로 vocab size $V$가 커야한다. 만일 $E \equiv H$라면, $H$가 커질수록 embedding matrix $E \times V$가 커지게된다.
이로 인해 모델은 수십억개의 파라미터를 갖게되고, 대부분은 학습동안 sparse하게 업데이트 될 것이다.

따라서 ALBERT는 embedding parameter를 두 개의 작은 matrix로 분해한다.
One-hot vector를 직접 크기 $H$의 hidden space로 projection하는 대신, 크기 $E$의 lower dimensional embedding space로 먼저 projection하고, 이후 hidden space로 한번 더 projection한다.
이러한 분해를 통해 embedding parameters를 $O(V \times H)$에서 $O(V \times E + E \times H)$로 줄일 수 있게된다.
이는 특히 $H \gg E$일 때 더욱 중요하다.

ALBERT는 모든 word pieces에 대해 같은 $E$를 사용한다. 
이는 전체 단어의 embedding에 비해 훨씬 문서들 사이에 더 균등하게 분포되어있고, 다른 단어에 대해 다른 embedding size를 갖는 것이 중요하기 때문이다 (Grave et al. (2017); Baevski & Auli (2018); Dai et al. (2019)).

**각 단어가 같은 사이즈의 임베딩**을 갖는건 아주 당연한 상식같다. 
하지만 사전연구를 얼추 살펴보았을 때 **단어 빈도수에 따라 임베딩 사이즈를 늘리는** 등 **각 단어마다 다른 사이즈의 임베딩**을 이용하는 연구들도 있기 때문에 본 연구에서는 이를 사용하지 않는다는 뜻으로 보인다.
다만 Transformer-XL은 이와 관련이 없는 것 같은데 무슨 이유로 들어갔는지 잘 모르겠다.

**[Adaptive input representations (Baevski & Auli, 2018)](https://arxiv.org/pdf/1809.10853.pdf)의 architecture**
{: .text-center}

![image](https://user-images.githubusercontent.com/47516855/168437979-36b97d01-9a39-407d-9f7e-e6737b51bcd9.png){: .align-center}{: width="600"}


**Cross-layer parameter sharing**

두번째 주요 변경점은 Cross-layer parameter sharing이다.
ALBERT에선 파라미터 효율을 위한 방법으로 cross-layer parameter sharing를 제안한다.
파라미터를 공유하는 방법에는 여러가지 방법이 있는데, 이는 레이어 간 feed-forward network (FFN)만 공유하거나, attention paramter만 공유하는 방법이다.
여기서는 **레이어 간 모든 파라미터**를 공유한다.
따로 명시하지 않는 이상 이러한 방법을 디폴트로 사용한다.

이와 비슷한 전략으로는 Universal Transformer(UT)과 Deep Equilibrium Models(DQE)가 있다.
ALBERT의 결과와는 달리 UT는 vanilla Transformer의 성능을 압도하였다.

DQE는 implict layer의 일종으로, **deep neural network를 단 하나의 레이어**를 통해 표현하는 모델이다.
즉, 모든 레이어는 동일한 파라미터를 갖으며, 무한히 깊은 레이어를 쌓았을 때, 레이어의 인풋이 아웃풋과 동일해지는 지점이 있다는 것이다.
이를 DQE는 평형점(equilibrium point)이라하며, 일반적인 네트워크를 학습시키는 대신 이 평형점을 직접 찾아내는 것을 목표로 한다.

![image](https://user-images.githubusercontent.com/47516855/168438937-ef12201e-eff5-4403-8f5f-4a33c6515d1f.png)

DQE는 자연어처리 등 다양한 모델에서 실험하여 파라미터의 양은 훨씬 적으면서 거의 비슷한 성능을 내었다. 

ALBERT에서는 DQE에서 **input/output embedding이 동일한**것과 다르게, L2 distances와 cosine similarity가 0으로 수렴하는 대신 진동하는 것을 찾아내었다.

![image](https://user-images.githubusercontent.com/47516855/168418714-881fb7f3-7332-4a17-a3e2-3c24a881109b.png){: .align-center}{: width="600"}

위 그림은 각 레이어의 input/output embedding에 대한 L2 distances와 cosine similarity 값을 나타낸 것이다.
여기서 두 레이어의 값을 비교하기 위해 L2 distance를 사용한 것을 볼 수 있다.
만일 두 레이어의 값이 같다면 L2 distance는 0으로 수렴해야 할 것이다.

그러나 ALBERT는 수렴하는 대신 특정 값으로 진동하는 것을 확인하였다.
또한, 레이어를 이동할 때 마다 ALBERT가 BERT보다 훨씬 더 부드러운 것을 볼 수 있는데, 이를 통해 weight-sharing이 **파라미터의 안정화**에 영향을 미치는 것을 알 수 있다 (특정 값으로 수렴).
비록 BERT와 비교했을 때 초반에 급격하게 하락하는 모습을 보였으나, 그럼에도 불구하고 24레이어가 지날 때 까지 0에 수렴하지 않는 것을 확인할 수 있다.
이는 ALBERT 파라미터의 solution space가 DQE의 solution space와는 매우 다르다는 것을 보여준다.

**Inter-sentence coherence loss**

BERT는 MLM과 NSP를 loss로 사용한다.
NSP는 문장 쌍의 관계를 추론하는 NLI와 같은 downstream task에서 사용하기 위해 도입되었다.
그러나 XLNet과 RoBERTa는 NSP의 영향이 없는 것을 밝혀냈고, 이를 제거하였다.

본 논문에서는 NSP의 효과가 없는 이유가 MLM과는 다르게 어렵지 않은 task 때문이라고 짐작한다.
NSP는 **topic prediction**과 **coherence prediction**을 하나의 task로 결합한 형태로 볼 수 있다.
BERT의 NSP를 수행할 때, negative-example의 경우 다른 문서에서 가져오기 때문에 topic과 coherence 측면에서 일치하지 않는다. 
그러나 topic prediction은 coherence prediction에 비해 학습하기 쉽고 MLM loss와 겹치는 부분이 있다.

본 논문의 저자들은 inter-sentence modeling이 language understanding의 중요한 측면 중 하나로 주장하지만, **coherence**에 더욱 기반을 둔 **Sentence-order prediction (SOP) loss**를 제안한다.
SOP loss는 topic prediction을 제거하는 대신 inter-sentence coherence를 모델링한다.

SOP loss는 positive examples (같은 문서에서 추출한 연속된 segment)는 BERT와 똑같은 테크닉을 사용하고, **negative examples은 똑같이 연속된 segment지만 순서가 뒤바뀐 것**을 사용한다.
이를 통해 모델이 discourse-level coherence properties에 대한 finer-grained distinctions을 학습하도록 한다.
추후 살펴보겠지만 이는 NSP는 SOP task를 풀 수 없으며 (즉, 더 쉬운 topic-prediction signal을 학습하며, SOP task에 대한 임의의 baseline 수준만 수행), 반면 SOP는 예측컨데 misaligned coherence cue를 분석하는 식으로 NSP task를 합리적인 수준에서 풀 수 있었다.
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




{: .align-center}{: width="600"}
