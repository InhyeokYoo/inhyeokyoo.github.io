---
title:  "MASS: Masked Sequence to Sequence Pre-training for Language Generation review"
toc: true
toc_sticky: true
permalink: /project/nlp/MASS-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2022-03-03
---

## 들어가며

MASS (**Ma**sked **S**equence to **S**equence Pre-training for Language Generation)는 난징대학교와 Microsoft에서 개발한 lanuge model로, 2019년 ICML에 소개되었다. MASS는 기존의 language model이 encoder/decoder only 모델을 쓰는 것과는 달리, seq2seq 구조를 이용하여 효율적인 language generation task를 진행할 수 있게 해준다. MASS를 이용하여 논문의 저자들은 low-resource 환경에서의 번역, 요약, 대화의 세가지 generation task에서 SOTA를 달성하였다. 

- [원문 보러가기](https://arxiv.org/pdf/1905.02450.pdf)
- [MASS repository 보러가기](https://github.com/microsoft/MASS)

## 1. Introduction

우리가 잘 알고있는 pre-training/fine-tuning 구조는 target에 대한 학습 데이터가 부족한 경우 널리 사용된다. Computer vision의 경우 매우 큰 스케일의 ImageNet으로부터 pre-train하고, object detection, segmantation 등등의 downstream task에 적용하기도 한다. 이로부터 영향을 받아 자연어처리 영역에서도 ELMo, GPT, BERT등의 방법론들이 인기를 끌었고, SOTA를 달성하였다.

그러나 language understanding과는 달리 language generation은 주어진 인풋에 조건부인 자연어 문장을 생성하는 것이 목적이다. 이에는 neural machine translation (NMT), conversational response generation, text summarization 등의 작업이 있다. Language generation tasks는 **일반적으로 데이터가 부족하며(data-hungry), low-resource이거나 심지어 zero-resource**인 경우도 흔하다. BERT와 같은 pre-training 방법론을 language generation에 바로 적용하는 것은 불가능한데, 이는 BERT가 NLU를 위해 설계되어있기 때문에 **오직 하나의 encoder/decoder**만 갖기 때문이다. 그러므로 seq2seq 구조를 갖는 generation task에는 pre-training 방법을 디자인하는 것이 매우 중요한 작업이라 할 수 있다.

이러한 이유들로 인해 본 저자들은 MASS(MAsked Sequence to Sequence learning)를 통해, generation에 어울리는 pre-training objective를 제안하고 있다. MASS는 앞서 언급했듯 seq2seq 구조를 갖는다. Encoder에서는 문장의 segment(연속적인 토큰)를 마스킹한 것을 입력으로 받고, decoder에서는 encoder에 attention하여 이를 예측하도록 한다.

BERT와 달리 encoder와 decoder 모두가 존재하기 때문에 다음의 두 단계를 통해 주의를 기울여 encoder와 decoder를 동시에 학습하도록 한다.
1. Encoder에서 마스킹한 부분을 예측해야 한다. Decoder가 이를 예측하도록 만들려면 MASS는 encoder로 하여금 **마스킹되지 않은 토큰을 이해하는 능력**을 부여해야 한다.
2. Encoder에서 마스킹되지 않은 부분은 decoder에서 마스킹하게 된다. 따라서 MASS는 decoder가 **이전 토큰**에 의존하기보다는 encoder에서 **정보를 뽑아**다 쓰게끔 만든다. 이를 통해 encoder와 decoder가 동시에 학습하도록 만든다.

저자들이 밝히는 주요 contribution은 다음과 같다.
1. language generation task에 효과적인 MASS를 제안함
2. NMT, 요약, 대화와 같은 language generation에서 상당한 성능 향상을 이끔

## 2. Related work

대부분의 language model들이 하는 이야기는 비슷하므로 (self-supervised, 데이터 양, etc.) MASS가 이전의 다른 모델과 갖는 상이함에 중점을 맞춰 리뷰해보자.

### 2.1. Sequence to Sequence Learning

Seq2seq은 AI분야에서 어려운 태스크로 여겨지며, NMT, 요약, 대화, QA 등의 다양한 genetation task를 다뤄왔다. Seq2seq은 딥러닝의 발전으로 최근 여러 연구자들의 관심을 받고 있으나 사용 가능한 데이터가 매우 적다는 단점이 있다. 따라서 그 무엇보다도 pre-training/fine-tuning 구조가 절실하며, 본 논문에서 초점을 맞추고 있는 것과 정확히 일치한다.

### 2.2. Pre-training for NLP tasks

앞서 언급했듯 이 부분은 최근 language model관련 논문이라면 한번쯤은 들을만한 이야기들로 적혀져있다. 
자연어처리 분야에선 pre-training을 사용하여 더 나은 language representation을 얻는 것을 목표로 한다. 
NLU에 경우 크게 **feature-based**와 **fine-tuning-based**로 나뉘어진다. 

Feature-based의 경우 pre-training을 이용하여 downstream task에서 사용할 representation과 feature를 얻는데 초점이 맞춰져 있다. 
대표적으로는 word2vec과 같은 word-level, doc2vec, Skip-thought vectors, Quick Thought과 같은 sentence-level의 representation, 마지막으로 ELMo, CoVe와 같이 context가 잘 반영된 feature를 잡는 representation이 있다.

Fine-tuning-based의 경우 우리가 잘 알고, 주류가 된 방법론으로, 모델을 pre-training한 후 supervised data에 대해 fine-tuning하는 형태로 되어있다.

또한, MASS와 같이 generation을 위해 pre-training encoder-decoder을 사용하는 연구들도 존재한다. Pre-training에 대한 연구의 서막을 열었던 Dai & Le (2015)와 [Ramachandran et al. (2016)](https://www.semanticscholar.org/paper/Unsupervised-Pretraining-for-Sequence-to-Sequence-Ramachandran-Liu/85f94d8098322f8130512b4c6c4627548ce4a6cc?p2df)의 연구같은 경우 auto-encoder를 사용하여 pre-training encoder-decoder를 학습하였다. 이를 통해 성능의 향상은 입증할 수 있었으나, 이는 제한되고, 일반화하기 어려웠으며, BERT처럼 가시적인 성과를 보이지는 못하였다. 
최근들어서는 XLM이 unsupervised NMT에서 성능향상을 이끌어 내었으나 encoder와 decoder가 분리되어 학습되기 때문에 이 둘 사이의 cross-attention을 이끌어 내지 못하였으며, 그 결과 seq2seq기반의 language generation에선 sub-optimal하다고 말할 수 있다.

이전 연구들과는 다르게 MASS는 이 둘을 동시에 학습하기 위해 심혈을 기울여 모델을 디자인했고, 대부분의 language generation task에 적용이 가능하다고 한다.

## 3. MASS

이번장에서는 sequence to sequence learning에 대해 간략하게 살펴본 뒤 본 논문에서 제안하는 MASS (MAsked Sequence to Sequence pre-training)에 대해 보도록 하자. 그다음에는 MASS와 이전에 제안되었던 BERT식의 masked language model과 일반적인 language model을 살펴본다. 

### 3.1. Sequence to Sequence Learning

$(x, y) \in (\mathcal X, \mathcal Y)$를 어떠한 문장 쌍이라 지칭하고, $x = (x _1, \cdots, x _m)$은 $m$개의 토큰을 갖는 source sentence, $y = (y _1, \cdots, y _n)$은 $n$개의 토큰을 갖는 target sentence라 하겠다. 
$\mathcal X, \mathcal Y$는 각각 source domain, target domain이라 하자.

Seq2seq 모델은 조건부 확률 $P(y \lvert x ; \theta)$를 추정하기 위해 파라미터 $\theta$를 학습하는 것을 목적으로 한다. 일반적으로는 log likelihood를 objective function으로 하며, 이는 $L(\theta ; (\mathcal X, \mathcal Y)) = \sum _{(x, y) \in (\mathcal X, \mathcal Y)} \log P(y \lvert x ; \theta)$로 표현한다.

조건부 확률 $P(y \lvert x ; \theta)$은 연쇄 법칙을 이용하여 factorize할 수 있고, 이는 $P(y \lvert x ; \theta) = \prod^n _{t=1} P(y _t \lvert y _{< t}, x; \theta)$가 된다.

Seq2seq의 주요 접근법은 encoder-decoder 구조로, encoder에선 source를 읽고 이에 대한 representation을 생성하고, decoder에선 이러한 representation에 조건부로 하여 target token을 생성한다. 이후에는 현재 토큰을 생성하기 위해 어느 source representation에 집중해야 하는지를 알아내는 attention mechanism이 소개되었다.

### 3.2. Masked Sequence to Sequence Pre-training

어떠한 source sentence $x \in \mathcal X$를 살펴보자. Notation $x ^{\setminus u:v}$는 $x$의 일부분으로, $u$부터 $v$까지의 토큰이 마스킹된 것으로 사용한다. 따라서 토큰 길이 $m$에 대해 $0 < u < v < m$이 성립하고, $k=u-v+1$로, 마스킹된 길이를 의미한다. 각 마스킹된 토큰은 $[\mathbb M]$으로 표현하며, 이 길이는 변화하지 않는다. $x ^{u:v}$는 $x$의 $u$부터 $v$까지의 fragment를 의미한다.

MASS는 seq2seq을 $x ^{\setminus u:v}$는 $x$를 통해 $x ^{u:v}$를 예측하는 식으로 학습한다. MASS도 다른 모델들과 마찬가지로 log likelihood를 objective로 하여 학습한다

$$
\begin{align}
L(\theta; \mathcal X) &= \frac{1}{\lvert \mathcal X \rvert} \sum _{x \in \mathcal X} \log P (x^{u:v} \lvert x ^{\setminus u:v}; \theta) \\
& = \frac{1}{\lvert \mathcal X \rvert} \sum _{x \in \mathcal X} \log \prod ^v _{t=u}  P (x^{u:v} _t \lvert x^{u:v} _{< t},  x ^{\setminus u:v}; \theta)
\end{align} \tag{1}
$$

![image](https://user-images.githubusercontent.com/47516855/144746608-7851f009-4455-4dec-8d88-a18fbb682730.png){: .align-center}{: height="400"}

위 그림과 같이 input sequence가 8개이고, 이의 fragement가 $x _3 x _4 x _5 x _6$가 마스킹되었다고 하자. 모델은 마스킹된 $x _3 x _4 x _5 x _6$을 $x _3 x _4 x _5$만 주어진채로 예측한다. 디코더는 다른 위치에 대해서는 (즉, 인코더에서 마스킹하지 않은) $[\mathbb M]$을 인풋으로 취한다. 본 방법론은 어떠한 뉴럴네트워크 기반의 인코더-디코더 프레임워크에서도 동작하므로, seq2seq에서 가장 성능이 좋은 트랜스포머로 특정하여 실험을 진행하였다.

![image](https://user-images.githubusercontent.com/47516855/146947499-407818fb-8947-49da-a42c-614bf83b1777.png){: .align-center}{: height="800"}

![image](https://user-images.githubusercontent.com/47516855/146948230-2b297bed-388b-41d2-bccc-9265b57953f4.png){: .align-center}{: width="400"}


위 그림처럼, BERT나 GPT는 MASS의 특별한 경우라고 볼 수 있다. 마스킹 길이를 나타내는 $k$를 어떻게 잡느냐에 따라, MASS는 BERT나 GPT가 될 수 있다. 비록 인코더-디코더 구조를 취하고는 있으나, BERT의 경우 디코더에 인풋이 없으므로 단순한 non-linear classifier로 취급할 수 있다. GPT의 경우에도 이와 비슷하다.

### 3.3. Discussions

비록 MASS가 이전의 GPT, BERT같은 모델과 연관이 있더라도, 다음과 같은 점에서 차이점이 있다고 할 수 있다.
- BERT나 다른 LM이 인코더, 디코더를 따로 학습하며, NLG에선 그렇게 좋은 성능을 내지 못한다는 점
- MASS는 인코더와 디코더를 동시에 학습.
  1. Seq2seq구조를 통해, 마스킹을 복원하는 과정에서 인코더로 하여금 마스킹되지 않은 토큰을 이해하며, 디코더로부터 이러한 정보를 끌어오도록 만듬
  2. 디코더로 하여금 연속적인 토큰을 예측함으로서 이산적인 토큰을 예측하는 것보다 더 나은 LM을 만들게 함
  3. 인코더에서 마스킹하지 않은 토큰을 디코더에서 마스킹하는 것이 디코더로 하여금 이전 토큰의 정보가 아닌 더 좋은 정보를 인코더에서 끌어오게끔 함

## 4. Experiments and Results

### 4.1. MASS Pre-training

#### Model Configuration

- Transformer
  - 6-layer encoder/decoder (1024 embedding)
  - hidden size (1024 embedding)
  - feed-forward (4096)
- NMT의 경우 단일 source/target 언어로 학습
  - 영어-불어, 영어-독어, 영어-루마니아어 데이터로 진행
  - XLM기반의 코드베이스로 진행
- 다른 generation의 경우 영어로만 학습

#### Datasets

- 2007년부터 2017년의 WMT News Crawl datasets에서 추출한 monolingual data
  - 190M(English), 62M(French), 270M(German) 문장으로 구성
- 루마니아어의 경우 pre-train단계에서 low-resource 환경을 테스트함
  - WMT16 data를 augment하여 2.9M의 문장을 만듬
- 길이가 175이상이면 제거
- source와 target에 대해 동시에 60000개의 subword를 만들어 학습

#### Pre-Training Details

- 연속된 토큰을 $[\mathbb M]$으로 치환
- start position $u$는 임의로 설정
- 토큰의 변환 확률은 BERT를 따름
- fragment length $k$는 대략 문장 내 토큰의 50%로 설정
  - 또한, 다양한 $k$를 설정하고 학습
- comutational cost를 위해 디코더에서 padding을 제거하지만 (masked tokens), 마스킹 되지 않은 토큰의 positional embedding은 그대로 유지
  - e.g. 처음 두 개의 토큰이 마스킹될 경우 제거되고, 세번째 토큰의 위치는 0이 아니라 3으로 지정
  - 비슷한 정확도에 효율은 50% 증가
- Adam optimizer with a learning rate of $10^{-4}$ for the pre-training
-  8 NVIDIA V100 GPU 
- 각 mini-batch마다 3천개의 토큰이 들어감

MASS의 효율을 알아보기 위해 NMT, text summarization and conversational response generation에 대해 실험한다. 
low-resource 시나리오 또한 시뮬레이션하였고, NMT에서는 zero-resource (unsupervised) setting으로 실험하였다.

### 4.2. Fine-Tuning on NMT

#### Experimental Setting

- unsupervised NMT의 경우 pre-trained model을 fine-tune할 bilingual data가 없으므로, pre-train 단계에서 사용한 monolingual data 활용
- 학습시 back-translation을 이용하여 pseudo bilingual data를 생성. 이때 denoising auto-encoder는 사용하지 않는다.
-  Adam optimize with initial learning rate $10^{−4}$, the batch size is set as 2000 tokens for each GPU
- evaluation 동안에는 English-French의 경우 *newstest2014*를, English-Romanian/English-German은 *newstest2016*에 대해 [multi-bleu.pl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl)을 이용하여 BLEU score를 계산함.

#### Results on Unsupervised NMT

![image](https://user-images.githubusercontent.com/47516855/148766211-f12f4e40-942c-4d46-8b8e-c65388b670e3.png){: .align-center}{: width="800"}

실험에 대한 결과는 위 Table 2에 잘 나와있다.
6개의 번역에 대해서 모두 좋은 성능을 내었다.
XLM (Lample & Conneau, 2019)이 SOTA인데, 이는 MLM (masked language model)과 CLM (causal language model)을 사용하는 BERT like pre-training를 인코더와 디코더에서 활용하였다.
MASS의 경우 이보다 더 좋은 것으로 나타났다

#### Compared with Other Pre-training Methods

이번 장에선 NLG에 대해 MASS와 다른 PLM을 비교한다.
베이스라인은 *BERT+LM*으로, MLM으로 인코더를 학습한 후 디코더에서 일반적인 LM을 학습한 것이다.
두번째 베이스라인은 *DAE*로, 단순하게 인코더와 디코를 denoising auto-encoder로 학습한 것이다.
논문에선 이 둘을 (1) pre-train한 후 (2) XLM(DAE loss + back-translation)과 같은 fine-tunin 전략으로 (3) unsupervised translation pairs로 학습하였다.
이 둘 모두 6-layer Transformer를 사용하였다.

![image](https://user-images.githubusercontent.com/47516855/153705289-8069550a-9b60-4f15-b564-637bfff13472.png){: .align-center}{: width="500"}

위 Table 3에서 볼 수 있듯, BERT+LM이 DAE를 능가하는 성능을 보였고, MASS는 BERT+LM을 능가하였다.
DAE는 임의의 토큰들 혹은 인접한 토큰들을 마스킹하는 denoising 테크닉을 사용한다. 그러나 대부분의 인코더-디코더 어텐션 구조에서는 residual connection을 사용하기 때문에 디코더의 최상위 레이어에서는 인코더의 토큰 임베딩에 **직접적으로 연결**된다.
따라서 이를 통해 마스크 되지 않은 토큰을 쉽게 학습한다.
반면, DAE의 디코더에서는 모든 문장을 인풋으로 받기 때문에 LM처럼 다음 토큰을 예측하기 충분한 정보를 갖고 있다.
따라서 **인코더에서 유의미한 정보를 추출**하기가 상대적으로 어렵다. 

#### Experiments on Low-Resource NMT

학습데이터가 부족한 low-resource 상황을 살펴보자.
WMT14 영어-프랑스어에서는 10K의, WMT16 영어-독일어에서는 100K, WMT16 영어-루마니아어에서는 1M개의 문장쌍을 뽑아 실험을 진행한다.
pre-training 단계에서 이전과 동일한 BPE를 사용하였고, 20,000스텝, Adam optimizer, $10^{-4}$의 학습률로 fine-tuning을 진행하였다. 
평가는 이전의 unsupervised NMT 결과와 동일한 데이터를 사용하여 진행한다.

![image](https://user-images.githubusercontent.com/47516855/153705793-2b5d56e2-2444-4983-8a5d-0fb0e3a390a6.png){: .align-center}{: width="900"}

위 Figure 3는 이에 대한 결과로, MASS가 학습데이터가 부족한 시나리오에서도 잘 동작함을 의미한다.

### 4.3. Fine-Tuning on Text Summarization

#### Experiment Setting

요약 태스크에서 PLM은 10K, 100K, 1M, 3.8M의 Gigaword corpus로 fine-tuning한다.
Gigaword corpus는 기사 본문과 제목으로 이루어져있고, 본문은 인코더에, 제목은 디코더에 넣어 요약 태스크를 학습한다.
성능의 평가지표로는 ROUGE-1, ROUGE-2, ROUGE-L의 F1 score를 사용한다.

#### Results

![image](https://user-images.githubusercontent.com/47516855/153706411-13f43263-c78c-4b79-9b60-099745f42ae6.png){: .align-center}{: width="400"}

Figure 4는 성능평가 결과이다.
MASS는 일관적으로 베이스라인 성능을 압도하였으며, 이를 통해 MASS가 다양한 스케일에 대한 학습데이터가 부족한 시나리오에서도 효과적임을 증명하였다.

#### Compared with Other Pre-Training Methods

앞장과 마찬가지로 *BERT+LM*과 *DAE*에 대해서도 성능 비교를 진행한다.
데이터는 3.8M의 요약데이터를 사용하였으며, 아래 표와 같이 좋은 성능을 보이고있다.

![image](https://user-images.githubusercontent.com/47516855/153706494-9ef33653-a7b0-4d27-aafb-6b675c72ba2d.png){: .align-center}{: width="400"}

### 4.4. Fine-Tuning on Conversational Response Generation

#### Experimental Setting

Conversational response generation는 대화에서 유연한 응답을 생성하는 태스크이다.
여기서는 140K 이상의 데이터를 갖는 Cornell movie dialog corpus를 이용하였고, 무작위로 10K/20K 쌍을 뽑아 validation/test set으로 사용하였다.
그 외 학습조건은 이전과 같고, perplexity를 사용하여 성능을 평가한다.

#### Results

실험은 임의로 선택한 10K쌍과 전체 110K쌍에 대해 진행하였다.
아래의 Table 5에서 확인할 수 있듯 베이스라인보다 낮은 PPL을 보였다.

![image](https://user-images.githubusercontent.com/47516855/153708553-d43e7a26-c54c-4748-9d24-c2e33d53fdd8.png){: .align-center}{: width="500"}

#### Compared with Other Pre-Training Method

마찬가지로 *BERT+LM*과 *DAE*에 대해 비교를 진행하였다.
이 결과 또한 위의 Table 5에서 확인할 수 있다.

### 4.5. Analysis of MASS

#### Study of Different k

마스킹(maksed fragment)의 길이 $k$는 중요한 하이퍼 파라미터로, 이에 따른 성능의 변화를 살펴보자.
$k$는 전체 문장의 10%부터 90%까지 10% 단위로 진행하였으며, 추가로 $k=1$ (BERT)과 $k=m$ (GPT)을 살펴보았다.

![image](https://user-images.githubusercontent.com/47516855/153708761-fa672fa7-7669-4cfd-b580-f4c684df6fd7.png){: .align-center}{: width="800"}

첫번째로 살펴볼 것은 영어-프랑스어의 pre-training 모델이다. WMT의 newstest2013을 validation set으로 사용하였으며, 이에 따른 PPL은 Figure 5a(영어)와 5b(프랑스어)에 나와있다. 이는 pre-trained model이 k가 50%-70% 사이일 때 최고의 validation PPL을 보이고 있다. 

Fine-tuning에서의 성능은 비지도 영어-프랑스어 번역에 대한 validation BLEU scores와 (Figure 5c), 텍스트 요약에서의 validation ROUGE scores (Figure 5d), conversational response generation에서의 validation PPL (Figure 5e)에서 확인할 수 있다.
MASS는 약 50%의 마스킹을 진행하였을 때 최고의 성능을 내는 것을 확인하였다.

$k=50$%는 인코더-디코더 사이의 균형을 잘 맞춰주는 값으로, 인코더나 디코더 내에서 **마스킹을 너무 많이** 한다면, 모델에 bias가 생겨 **마스킹이 덜 되는 인코더/디코더에 정보를 얻기 위해 의존**할 것이고, 이는 인코더-디코더 구조를 잘 이용해야하는 NLG 태스크에서 적절하지 않게된다. 극단적인 $k=1$ (BERT)나 $k=m$ (GPT) 둘다에서도 NLG는 좋은 성능을 보이지는 못하였다.

#### Ablation Study of MASS

MASS에서 제안하는 masked sequence to sequence pre-training 방법은 신중을 기해 다음의 두 가지를 디자인하였다.
1. 인코더쪽에서 연속된 토큰을 마스킹하고 이를 디코더에서 예측한다. 이를 통해 개개의 토큰을 예측하는 것 보다 더 나은 language modeling capability를 얻는다.
2. 인코더에서 마스킹되지 않았던 토큰을 디코더에서 마스킹한다. 이를 통해 이전 토큰에서의 정보를 이용하기보단 디코더가 인코더에서 정보를 추출하도록 만든다.

첫번째 디자인에 대해서는 임의로 개개의 토큰을 마스킹하여 연속된 토큰을 마스킹하는 것의 성능을 알아보고 (Table 6의 *Discrete*), 두번째 디자인은 인코더에서 마스킹하지 않았던 토큰을 디코더에서 마스킹하는 대신 전부 집어넣는 방법이다 (Table 6의 *Feed*).

![image](https://user-images.githubusercontent.com/47516855/153738927-d6f1bc52-fa31-4d9a-ba8f-29837e81a73f.png){: .align-center}{: width="500"}

이들 모두 비지도 영어-프랑스어 번역에 대해 테스트하였으며, 이 결과 MASS의 유용함을 알 수 있다.