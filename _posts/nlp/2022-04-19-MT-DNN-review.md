---
title:  "Multi-Task Deep Neural Networks for Natural Language Understanding review"
toc: true
toc_sticky: true
permalink: /project/nlp/MT-DNN-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
  - Multi-task learning
use_math: true
last_modified_at: 2022-05-08
---

## 들어가며

MT-DNN은 Microsoft에서 개발한 language model로, multi-task learning과 language model pre-training을 결합한 모델이다.
2019년에 ACL에서 accept되었으며, [이곳](https://aclanthology.org/P19-1441/)에서 논문을 확인할 수 있다.

최근들어 multi-task learning(MTL)을 적용하는 딥러닝 모델이 각광받고 있다 (Collobert et al., 2011; Liu et al., 2015; Luong et al., 2015; Xu et al., 2018; Guo et al., 2018; Ruder12 et al., 2019).
이에 대한 이유로는 다음과 같은 두 가지 이유가 있다.
1. 지도학습의 경우 대량의 레이블 데이터가 필요한데, 이는 현실에서 충족하기 어렵다. MTL은 많은 연관된 데이터를 통해 이를 효율적으로 충족시켜 지도학습의 이점을 살릴 수 있다.
2. MTL을 사용하여 특정 태스크에 대한 overfitting을 방지하는 **정규화** 효과가 있다. 따라서 태스크를 넘나드는 universal representation을 얻을 수 있다.

이 중 MTL과 정규화에 대한 이야기는 처음 듣는데, 아래 위키피디아를 살펴보면 다음과 같이 소개하고 있다.

> Multi-task learning works because regularization induced by requiring an algorithm to perform well on a related task can be superior to regularization that prevents overfitting by penalizing all complexity uniformly.

본 논문에서는 MTL과 language model(LM)이 서로 **상호보완적**인 기술이라 주장하고 있으며, 이를 결합하여 텍스트의 표현력을 향상시킬 수 있다고 이야기한다.

이를 달성하기 위해 MT-DNN은 과거 [Liu et al. (2015)](https://aclanthology.org/N15-1092/)에서 제안한 MT-DNN 구조의 shared text encoding layers를 BERT로 대체하는 방안을 제안한다.

![image](https://user-images.githubusercontent.com/47516855/164012624-5f9fb828-5f05-459d-a7bd-a2f85b9beab2.png){: .align-center}{: width="600"}

위 그림에서 볼 수 있듯 아랫단의 레이어(i.e., text encoding layers)는 **모든 태스크간에 공유**하며, 상위레이어는 **태스크에 따라** 다르게 갖는다.
BERT와 비슷하게 fine-tuning을 통해 특정 태스크를 학습시킬 수 있지만, pre-training과정에서 MTL을 이용한다는 점이 다르다.

MT-DNN은 9개의 GLUE 데이터 중 8개의 데이터에서 SOTA를 달성하였으며, BERT보다 2.2%p 상승한 82.7%의 결과를 보였다.
또한 SNLI와 SciTail로 확장하여 MT-DNN의 우월성을 보였다.
MT-DNN을 통해 얻은 representation은 대체로 BERT보다 작은 수의 in-domain 레이블 데이터로 domain adaptation이 가능하다.

일례로 MT-DNN에 domation adaptation을 적용한 모델은 SNLI에서 이전 SOTA보다 1.5%p 높은 91.6%, SciTail에서는 6.7%p 높은 95.0%를 달성하였다.
심지어는 원본 데이터의 0.1%/1.0%만 사용하여도 현존하는 모델보다 더 좋은 성능을 보였다.
본 결과는 MTL을 통한 MT-DNN의 **특별한 일반화 능력**을 증명한 것이라 볼 수 있다.

## Tasks

MT-DNN은 single-sentence classification(CoLA, SST-2), pairwise text classification(RTE, MNLI, QQP, MRPC), text similarity scoring(STS-B), relevance ranking(QNLI)의 네가지 형태의 NLU 태스크를 결합하였다.

여기서 QNLI의 경우 약간의 변형을 가하였다. 
기존의 QNLI가 주어진 질문에 대한 답변으로 적합한지를 판단하는 binary classification인 반면, 여기서는 pairwise ranking task로 변형하여 모델이 candidate를 랭킹하고 정답이 상위권에 랭킹되었는지를 판별하는 것이다.
이를 통해 기존의 binary classification보다 정확도를 크게 향상시켰음을 보일 것이다.

## The Proposed MT-DNN Model

![image](https://user-images.githubusercontent.com/47516855/164012624-5f9fb828-5f05-459d-a7bd-a2f85b9beab2.png){: .align-center}{: width="600"}

**Lexicon Encoder ($l _1$)**

MT-DNN의 인풋 $X=\\{x _1, \cdots, x _m \\}$는 m의 길이를 같는 토큰의 시퀀스이다. 
첫 토큰 $x _1$은 항상 `[CLS]`가 된다.
만일 $X$가 문장 쌍 $(X _1, X _2)$로 이루어져있다면, BERT랑 동일하게 `[SEP]`로 분리한다.
각 토큰 벡터는 이에 해당하는 word, segment, positional embeddings의 합으로 이루어진다.

**Transformer Encoder ($l _2$)**

Transformer encoder를 사용하여 input representation vectors($l _1$)를 contextual embedding vectors의 시퀀스 $\mathbf C \in \mathbb R^{d \times m}$로 맵핑한다.
이는 다른 태스크에 대해 공유된다.

이후에는 NLU와 GLUE에 대한 task specific layer를 설명한다.

**Single-Sentence Classification Output:**

$\mathbf x$를 `[CLS]` 토큰의 contextual embedding($l _2$)라 하자.
SST-2의 경우 $X$가 특정 레이블 $c$(i.e., the sentiment)로 분류될 확률은 softmax를 통한 logistic regression으로 얻어진다.

$$
P _r(c \rvert X) = \text{softmax}(\mathbf W^{\intercal} _{SST} \cdot \mathbf x) \tag{1}
$$

**Text Similarity Output:**

STS-B의 경우 $\mathbf x$를 문장 쌍 $(X _1, X _2)$의 semantic representation으로 볼 수 있다.
그러면 다음과 같이 유사도를 비교할 수 있다.

$$
\text{Sim}(X _1, X _2) = \mathbf w^{\intercal} \cdot \mathbf x \tag{2}
$$

$\text{Sim}(X _1, X _2)$는 $(-\infty, \infty)$ 사이의 값을 갖는다.

**Pairwise Text Classification Output:**

여기서 NLI는 전제(premise) $P = (p _1, \cdots, p _m)$와 가설(hypothesis) $H = (h _1, \cdots, h _n)$ 사이의 논리적 연관성 $R$을 찾는 것으로 정의한다 (다만 기존과도 크게 다르지 않다. 원래 NLI는 주어진 전제에 대해 가설이 진실인지(entailment), 가짜인지(contradiction), 정해지지 않았는지(neutral) 찾는 것이다).

여기서의 output module은 stochastic answer network (SAN)의 answer module을 따른다.
SAN의 answer module은 multi-step reasoning을 이용한다.
주어진 인풋을 직접적으로 예측하기보다, state를 유지하고 반복하여 예측값을 정제한다.
SAN에 대한 자세한 설명은 [이곳](https://eagle705.github.io/articles/2019-10/SAN_for_NLI)을 참고하자.

SAN의 answer module이 동작하는 방식은 다음과 같다.

먼저 전제 $P$에 존재하는 단어의 contextual embeddings을 concat하여 $P$에 대한 working memory를 만든다.
이는 트랜스포머 인코더의 아웃풋으로 $\mathbf M^p \in \mathbb R^{d \times m}$으로 표현한다.
가설 $H$에 대한 working memory도 이와 비슷하며 $\mathbf M^h \in \mathbb R^{d \times n}$로 표현한다.
그리고 얻어진 메모리에 대해 $K$-step reasoning을 수행하여 이들간의 관계를 예측한다.

Initial state $\mathbf s^0$는 $\mathbf s^0 = \sum _j \alpha _j \mathbf M^h _j$로 얻어지며, $\alpha _j = \frac{\exp (\mathbf w^{\intercal} _1 \cdot \mathbf M^h _j)}{\sum _i \exp (\mathbf w^{\intercal} _1 \cdot \mathbf M^h _i)}$이다.
$\\{1, 2, \cdots, K-1 \\}$의 범위를 갖는 time step $k$에서 state는 $\mathbf s^k = \text{GRU}(\mathbf s^{k-1}, \mathbf x^k)$로 정의된다.
여기서 $\mathbf x^k$는 이전 state $\mathbf s^{k-1}$과 memory $\mathbf M^p$로 정의되어 $\mathbf x^k = \sum _j \beta _j \mathbf M^p _j$가 되며, $\beta _j= \text{softmax}(\mathbf s^{k-1} \mathbf W^{\intercal} _2 \mathbf M ^p)$가 된다.

One-layer classifier는 각 time step $k$에서 관계를 판별하는데 사용되며, 다음과 같다.

$$
P^k _r = \text{softmax}(\mathbf W^{\intercal} _3 [\mathbf s^k; \mathbf x^k; \lvert \mathbf s^k - \mathbf x^k \rvert; \mathbf s^k \cdot \mathbf x^k]) \tag{3}
$$

마지막으로 모든 $K$개의 아웃풋을 평균낸다

$$
P _r = \text{AVG}([P^0 _r, \cdots, P^{K-1} _r]) \tag{4}
$$

각 $P _r$은 모든 관계 $R \in \mathcal R$에 대한 확률분포이며, 평균내기전에 **stochastic prediction dropout**를 사용한다.
*디코딩* 시에는 모든 아웃풋을 평균내어 강건성을 향상시킨다 (여기서 디코딩이라 했지만, 문맥상 inference를 의미하는 것으로 보인다).

stochastic prediction dropout은 하나의 아웃풋을 사용하기보단 multiple step reasoning의 모든 스텝에서의 아웃풋을 이용하는 것이다.
이를 통해 직관적으로는 특정 스텝의 예측에 지나치게 강조하는 **step bias problem**을 피할 수 있고, 모델로 하여금 매 스텝마다 좋은 결과를 낼 수 있게한다.

여태까지의 복잡한 프로세스를 그림으로 나타내면 다음과 같다.

![image](https://user-images.githubusercontent.com/47516855/167287269-18424840-350f-457b-b01a-97d2beb8801a.png){: .align-center}{: width="700"}


**Relevance Ranking Output:**

QNLI를 예로 들어보자. 위와 마찬가지로 $\mathbf x$가 `[CLS]` 토큰의 임베딩이라 할 때, 이는 질문과 후보답변 $(Q, A)$를 담고있다.
따라서 이 둘의 관계는 다음과 같이 정의된다.

$$
\text{Rel}(Q, A) = g(\mathbf w^{\intercal} _{\text{QNLI}} \cdot \mathbf x)
$$

주어진 $Q$에 대해 모든 답변 후보를 이에 따라 랭킹한다.


### The Training Procedure

MT-DNN의 학습은 pre-training과 multi-task learning의 두 단계로 이루어진다.
Pre-training 시에는 BERT를 사용한다.
MTL에는 Algorithm 1을 따라 학습한다.

![image](https://user-images.githubusercontent.com/47516855/165088000-6aa3131b-38bd-4b8d-8d5d-d50a9ef73953.png){: .align-center}{: width="400"}

분류 문제(single-sentence, pairwise text classification)에서는 다음과 같은 CE loss를 사용한다:

$$
-\sum _c \mathbb{1}(X, c) \log(P _r (c \lvert X)) \tag{6}
$$

여기서 $P _r(.)$은 식 (1)이나 (4)를 의미한다.

Text similarity scoring(STS-B)는 다음과 같은 objective를 사용한다.

$$
(y - \text{Sim}(X _1, X _2))^2 \tag{7}
$$

$\text{Sim}(.)$은 식 (2)를 의미한다.

Relevance ranking tasks(QNLI)의 경우 [Burges et al., 2005](https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf); [Huang et al., 2013](https://dl.acm.org/doi/10.1145/2505515.2505665)를 따른 pairwise learning-to-rank paradigm을 사용한다.

질문 $Q$가 주어질 때, 정답 후보군 $\mathcal A$에는 positive example $A^+$와 $\lvert \mathcal A \rvert - 1$개의 negative example이 있다.
이에 대해 NLL loss를 적용하여 학습시킨다.

$$
- \sum _{(Q, A^+)} P _r (A^+ \lvert Q) \tag{8}
$$

$$
P _r (A^+ \lvert Q) = \frac{\exp (\gamma \text{Rel} (Q, A^+))}{\sum _{A' \in \mathcal A} \exp (\gamma \text{Rel} (Q, A'))} \tag{9}
$$

$\text{Rel}$은 식 (5)에서 정의되어있고, $\gamma$는 tuning factor로, held-out data를 사용하여 결정한다.
여기서는 단순하게 1로 세팅한다.

## Experiments

MT-DNN의 성능을 평가하기 위해 GLUE, SNLI, SciTail의 세 가지 NLU 벤치마크 데이터에 대해 실험해본다.
이들에 대한 요약은 아래 Table 1에 나와있다.

![image](https://user-images.githubusercontent.com/47516855/165094623-691792fd-d9d2-4fd9-835c-a4efe6d02ec5.png){: .align-center}{: width="600"}

SNLI와 SciTail의 경우 domain adaptation에만 사용하며, MTL/MTL+fine-tuning을 통해 MTL의 성능을 증명하였다.


### Implementation details

구현 디테일은 다음과 같다
- Optimizer: Adamax
- lr: 5e-5
- batch size: 32
- Max epoch: 5
- Scheduler: linear learning rate decay schedule with warm-up over 0.1
- dropout: task specific layers: 0.1,  MNLI: 0.3, CoLa: 0.05
- Gradient clipping 사용 (norm 1)
- wordpieces 사용
- 512 length

### GLUE Main Results

여기서 사용된 BERT large는 저자들이 공개한 모델이며, MT-DNN의 경우 *The Proposed MT-DNN Model*에서 설명한대로 학습하였다.
MT-DNN은 pre-trained BERT LARGE를 사용하였으며, MTL을 통해 모델을 학습하고, 각 GLUE에 대해 fine-tuning까지 진행한 모델이다.

다만 여기서 MTL과 fine-tuning을 동시에 해주었다는게 흥미로운데, 각 MTL/fine-tuning 학습시에 사용한 데이터가 동일한 것으로 보이기 때문이다.
즉, 동일한 데이터를 이용하여 구조의 변경만으로 성능의 향상을 이끌어낸 샘인데, 연구자의 이런 아이디어나 결과 모두 매우 흥미롭다.

![image](https://user-images.githubusercontent.com/47516855/165096492-2567bf0a-7dc7-486b-b2fe-d90ab509b158.png){: .align-center}{: width="800"}

Table 2를 보면 MT-DNN이 WNLI를 제외한 모든 태스크에서 능가하는 것으로 나타났다.
Baseline과 MT-DNN 모두 BERT-large를 사용하였으므로, MT-DNN의 **성능향상은 MTL에서 나타났다**고 볼 수 있다.

MTL은 특히나 적은 수의 in-domain 데이터가 있는 경우에 유용한데, 테이블에서 볼 수 있듯 **적은 in-domain을 갖는** RTE(NLI), MRPC(Paraphrase)가 같은 태스크인 MNLI(NLI), QQP(Paraphrase)보다 **성능 증가폭이 더 큰 것**을 볼 수 있다.

MT-DNN은 모든 GLUE에 대해 MTL을 사용하였기에 **fine-tuning을 적용하지 않고** 테스트를 하는 것도 가능하다.
저자들은 이에 대해서도 실험을 진행하였다.

**파인튜닝을 하지 않은 MT-DNN** CoLa를 제외한 모든 태스크에서 성능이 우월한 것으로 나타났다.
본 논문의 분석에 따르면 CoLa의 경우 어렵고, 적은 in-domain 데이터를 갖으며, 태스크와 데이터셋이 다른 GLUE와 다르게 독특하므로 MTL을 통해 얻는 이점이 작았을거라 한다.
따라서 결과적으로 MTL은 CoLa에서 **과소적합**하는 경향을 보였다.

이러한 경우에는 파인튜닝을 진행하면 성능을 향상시킬 수 있다.
파인튜닝을 진행하는 경우 58.9%에서 62.5%로 성능이 향상됨을 보인다.
이와 같은 사실들을 통해 MT-DNN이 생성하는 표현이 domain adaptation에 효과적임을 증명할 수 있다.

MT-DNN의 또 다른 이점은 유연한 모델 구조로, 독립적인 태스크에 적용하는 task-specific model과 training methods를 사용할 수 있다는 점이다.
앞서 보았던 SAN이 이에 대한 예시이며, 다음을 통해 이에 대한 분석을 진행한다.

Single-Task DNN(ST-DNN)은 MT-DNN에서 MTL을 제외한 것으로, MT-DNN의 design choice를 살펴보기 위해 학습하였다.
ST-DNN은 각 GLUE 태스크에 따로 학습되었으며, pairwise text classification tasks의 경우 SAN을 제외하면 BERT와 동일하다.

![image](https://user-images.githubusercontent.com/47516855/165096567-203f61ae-92a9-40e1-8c45-25976da3bd06.png){: .align-center}{: width="700"}

위 Table 3를 보면 모든 태스크에 대해 MT-DNN이 BERT보다 우월하며, 특히나 SAN의 효율에 대해 알 수 있다.
이에 추가로 QNLI를 보면 ST-DNN은 pairwise ranking loss를 쓰는반면, BERT는 이를 이진 분류와 CE로 학습한다.
이와 같은 결과를 통해 문제 정의의 중요성을 알 수 있다.

### Domain Adaptation Results on SNLI and SciTail

실용적인 시스템을 구축하는데 있어 중요한 기준 중 하나는 새로운 도메인과 태스크에 얼마나 빠르게 적응하는가이다.
이는 새로운 도메인이나 태스크에 대한 데이터를 모으는데 자원이 많이 소모되기 때문이며, 종종 데이터가 없거나 매우 제한된 데이터만 있는 경우도 존재한다.

이러한 관점에 맞춰 SNLI, SciTail 두 개의 NLI 태스크에 domain adaptation을 수행한다.
절차는 다음과 같다.

1. MT-DNN 혹은 BERT 모델을 초기 모델로 사용하고, base와 large 둘 다 사용한다.
2. 각 태스크에 대해 학습된 MT-DNN을 적응시켜 새로운 모델을 만든다.
3. 각 태스크의 테스트 셋을 통해 평가한다.

training/dev/test은 원본 그대로 사용하되, 학습의 경우 0.1%, 1%, 10%, 100%로 나누어 사용한다.
결과는 5번 실험한 것의 평균값을 사용한다. 
SNLI와 SciTail에서 학습 데이터 양에 따른 결과가 아래 그림과 표에 나타나있다.

![image](https://user-images.githubusercontent.com/47516855/165105829-3ba662e5-c650-4be9-8a37-7abc3a69e5bc.png){: .align-center}{: width="400"}

---

![image](https://user-images.githubusercontent.com/47516855/167286139-69abaec4-be0c-433f-a8d0-e8b431a7a392.png){: .align-center}{: width="400"}


이 결과 MT-DNN이 일관적으로 BERT보다 월등한 성능을 내며, **더 적은 수의 데이터를 사용하여 학습**할 경우 **MT-DNN와 BERT의 갭이 커지는** 것을 확인하였다.
이러한 결과는 MT-DNN을 통해 학습한 표현법이 **domain adaptation에 더욱 효과적**이라는 것을 시사한다.

이번에는 모든 in-domain training samples을 사용하는 것과 SOTA를 비교해보도록 한다.

![image](https://user-images.githubusercontent.com/47516855/167286241-63169fe5-d6c8-4ee6-a5e6-0966ab0c46a8.png){: .align-center}{: width="400"}

MT-DNN large는 SNLI 기존 SOTA에서 1.5%p 상승한 91.6%을 달성하였고, SciTail에서 6.7%p 상승한 95.0%을 달성하여 두 데이터셋에서 모두 SOTA를 달성하였다.
이를 통해 domain adaptation에서도 MT-DNN의 우월함을 알 수 있다.

## Conclusion

지금까지 MT-DNN을 살펴보았다. 
T5나 RoBERTa 같은 large model과는 시기적으로 차이가 있기 때문에 연구방향도 살짝 다른 것을 확인할 수 있었다.

같은 데이터임에도 불구하고 구조의 변경만으로 좋은 결과를 이끌어낸 직관과 QNLI를 다룰 때 task formulation을 변경하여 성능을 향상해낸 연구자들의 contribution이 매우 흥미로운 논문이었다.