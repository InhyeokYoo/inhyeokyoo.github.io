---
title:  "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks review"
toc: true
toc_sticky: true
permalink: /project/nlp/SBERT-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2022-05-31
---

## Introduction

- Clustering에서 왜 poly encoder가 문제인가
- Triplet loss 어디다 쓰는지?

BERT가 비록 다양한 sentence classification과 sentence-pairregression task에서 SOTA를 달성했지만, 대규모의 문장 쌍을 다룰 때 연산이 많아진다는 단점이 있다.
이로인해 클러스터링이라던가, IR, semantic similarity 비교 등의 태스크에서 엄청난 오버헤드가 일어난다.

SBERT는 2019년 ACL에 accept된 논문으로, 이러한 BERT의 단점을 siamese/triplet network를 이용하여 보완한다.
논문은 [이곳](https://arxiv.org/pdf/1908.10084.pdf)에서 확인할 수 있으며, sentence-transformers [라이브러리](https://www.sbert.net/)를 통해 이용할 수 있다 (또한 [Hugging face](https://huggingface.co/sentence-transformers)도 제공한다).

## Challenges

비록 BERTA와 RoBERTa가 Semantic Textual Similarity(STS)와 같은 문장 쌍으로 이루어진 regression task에서 SOTA를 달성하였으나, 두 개의 문장이 모델로 들어가게 되므로(cross encoder) **오버헤드가 발생**하게된다.

예를들어 BERT를 통해 $n=10000$ 문장에서 가장 유사한 두 개의 문장을 찾는다고 해보자.
이 경우 $n \dot (n-1)/2=49995000$ (약 50M)의 inference computations을 필요로한다.
이는 V100 GPU를 사용했을 때 약 65시간정도 소요되는 양이다.
이와 비슷하게 BERT를 통해 Quora에서 40M개 이상의 질문 중 어떤 질문과 가장 비슷한지 찾는 태스크(QQP) 또한 **50시간 이상의 시간**을 소요한다.  

이러한 오버헤드 뿐만 아니라 **성능**에서도 큰 이슈가 있다.
Clustering과 semantic search에서 주로 다루는 방법은 **각 문장을 vector space로 맵핑**하여 **의미적으로 비슷한 문장은 가깝게** 만드는 식이다.
BERT의 가장 큰 단점은 문장 하나에 대해서는 임베딩을 계산할 수 없기 때문에, BERT를 사용하게 되면 주로 `[CLS]`토큰을 사용하거나, output vector의 평균을 통해 문장을 임베딩하게 된다. 
그러나 이는 좋지 않은 방법으로 문장이 임베딩되며, 심지어는 **GloVE보다도 성능이 떨어지는 것**으로 나타났다.

## Contributions

SBERT는 BERT에 siamese/triplet network를 이용, *의미적으로 의미있는(semantically meaningful)* 문장 임베딩을 가능토록한다.
여기서 *의미적으로 의미있는* 이라는 뜻은 의미적으로 유사한 문장이 vector space 내에서 가깝다는 것을 의미한다.
SBERT를 통해 지금까지는 적용 불가능했던 large-scale의 semantic similarity 비교, 클러스터링, semantic search를 이용한 정보 검색 등을 가능토록 한다.

10,000개의 문장 쌍에 대한 임베딩의 경우 BERT가 65시간 걸렸던 반면 SBERT는 5초 이내로 계산할 수 있게되며, 코사인 유사도의 경우 0.01초 이내로 계산할 수 있게 된다.

성능 측면에선 STS의 경우 InferSent보다 11.7 포인트, Universal Sentence Encoder에 비해 5.5 포인트 높은 성능을 보였으며, SentEval의 경우 각 각 2.1/2.6 포인트 높은 성능을 달성하였다.

또한, 이전의 neural sentence embedding은 random initialization에서 학습을 시작하였는데, SBERT의 경우 pre-trained BERT/RoBERTa에 fine-tuning을 통해 의미있는 sentence embedding을 얻는다.
이를 통해 학습 시간을 상당부분 감소시킬 수 있다. Tuning에는 약 20분 미만이 걸렸으며, 다른 모델보다 더 좋은 결과를 얻었다.

## Method

SBERT에선 세 가지 pooling 방법을 비교한다
- `CLS`-token 사용
- [default] 모든 output vector에 대한 평균 (`MEAN`)
- output vector에 대한 max-over-time (`MAX`)

> ㅇㅇ
```

```
> ㅇㅇ

BERT/RoBERTa를 fine-tuning하기 위해서 FaceNet의 siamese/triplet network를 사용한다.
Siamese/triplet network는 가용가능한 학습 데이터에 의존하므로 다음과 같은 목적함수에 대해 실험을 진행한다.

### Classification Objective Function

다음은 classification에서 사용하는 objective function이다.

![image](https://user-images.githubusercontent.com/47516855/171989665-3dfcd891-44b5-4529-aeaa-887c61a6d566.png){: .align-center}{: width="300"}

Sentence embedding $\mathbf{u}, \mathbf{v}$와 이의 element-wise difference $\lvert \mathbf{u} - \mathbf{v} \rvert$를 concatenation하고, 이를 trainable weight $\mathbf{W} _t \in \mathbb{R}^{3n \times k}$와 곱하여 objective Function을 만든다.

$$
\mathbf{o} = \text{softmax}(\mathbf{W} _t (\mathbf{u}, \mathbf{v}, \lvert \mathbf{u} - \mathbf{v} \rvert))
$$

여기서 $n$은 embedding dimension이고 $k$는 label의 갯수이다.
이후 cross entropy loss를 사용하여 이를 최적화한다.

여기서 사용한 문장 임베딩간의 차이에 대해 concatenate하는 것은 MT-DNN에서도 본 것 같은데, 어떠한 이유로 다음과 같은 결과가 나왔는지 궁금하다.

본 function은 NLI task에서 사용한다.

### Regression Objective Function

다음은 regression task에서 사용하는 objective function이다.

![image](https://user-images.githubusercontent.com/47516855/172142599-a77e56d8-f32d-4e1f-86b7-f93dc70e9fc8.png){: .align-center}{: width="300"}

$\mathbf{u}, \mathbf{v}$간의 코사인 유사도가 위 그림과 같이 계산된다.
이에 대한 objective function으로 MSE loss를 사용한다.

본 function은 regression에서 활용한다.

### Triplet Objective Function

기준이 되는 anchor sentence $a$, anchor와 동일한 클래스인 positive sentence $p$, anchor와 다른 클래스인 negative sentence $n$에 대해 $a$와 $p$의 거리가 $a$와 $n$의 거리보다 가깝게 되는 triplet loss를 계산한다 (앞서 triplet network라 표현되었지만 엄밀하게는 triplet loss를 사용한다).
수학적으로는 다음과 같은 loss function을 최소화한다.

$$
\max(\| \mathbf{s _a} - \mathbf{s _p} \| - \| \mathbf{s _a} - \mathbf{s _n} \| + \epsilon, 0)
$$

여기서 $s _x$는 $a/n/p$에 대한 임베딩이며, $\| \cdot \|$은 distance metric (Euclidean distance 사용), $\epsilon$은 margin이다.
보다시피 FaceNet에서 사용한 loss와 동일하다.

여기서 일반적으로 사용하는 cosine similarity와 Euclidean distance의 차이는 [11-785 Introduction to Deep Learning Fall 2020에서 제공하는 Homework](https://deeplearning.cs.cmu.edu/F20/document/homework/Homework_2_2.pdf)를 보면 나와있다.


## Experiment

### Training Details

SBERT는 SNLI와 MNLI에 대해 학습된다.
SNLI는 570,000개의 문장 쌍이 주어지며, 레이블은 contradiction, eintailment, neutral로 주어진다.
MNLI는 430,000의 문장 쌍으로 구성되며, 대화부터 글까지 여러 장르를 포함한다.

SBERT는 한 epoch으로 3-way softmax-classifier objective function(label이 세 개인 것을 의미)을 통해 학습한다.
Batch size는 16으로 Adam optimizer와 learning rate 2e-5를 적용하였고, 학습 데이터의 10%에 linear learning rate warm-up를 적용하였다.

또한, Argument Facet Similarity (AFS) corpus에 대해서도 추가적으로 실험하였다.
AFS는 논란이 많은 총기 규제, 동성혼, 사형제도에 대해 소셜 미디어에서 수집한 6천여개의 문장(sentential argument)으로 구성되어있다.
데이터는 0점(완전히 다름)부터 5점(완전히 동일) 사이의 점수가 매겨져있다.

AFS corpus에서의 유사도는 STS에서의 유사도와 많이 다른데, STS의 경우 기술적인(descriptive) 반면 AFS는 대화로부터 논증적으로 발췌한 것이 때문이다.
AFS 논문에서는 STS와의 차이점을 다음과 같이 소개하고 있다.

> We distinguish AFS from STS because: (1) our data are so different: **STS data consists of descriptive sentences whereas our sentences are argumentative excerpts** from dialogs; and (2) our definition of facet allows for sentences that express opposite stance to be realizations of the same facet (AFS = 3) in Fig. 10.

논쟁이 비슷하려면 

### Evaluation - Semantic Textual Similarity

일반적인 방법론은 복잡한 regression function을 학습하여 sentence embedding과 유사도 사이의 맵핑을 가능케한다.
그러나 이러한 regression function은 문장 쌍으로 동작하고, 이들의 조합이 너무나 많기 때문에 scalable하기가 쉽지 않다.

대신 SBERT는 cosine-similarity를 사용하여 두 문장간의 유사도를 비교한다.
다른 distance metric인 negative Manhatten과 negative Euclidean distance도 실험하였지만 코사인 유사도와 비슷한 성능을 내었다.

### Unsupervised STS

본 실험은 STS에 학습하지 않은채로 STS에 테스트를 진행한 결과를 보여준다.
실험에서 사용한 데이터는 STS tasks 2012 - 2016와 SICK이다.
STS와 SICK 모두 문장 쌍 간의 의미적 유사도를 0에서 5사이로 표현한다.

피어슨 상관계수의 경우 STS에서 사용하기가 좋지 않다.
따라서 문장 임베딩과 레이블간의 코사인 유사도에 대해 스피어만 상관계수를 통해 성능을 측정한다.

![image](https://user-images.githubusercontent.com/47516855/172149711-ccf1baeb-b02d-4125-8ae7-df7bf268fda2.png){: .align-center}{: width="700"}

BERT를 그대로 사용하는 것이 제일 안 좋았으며, GloVe보다도 성능이 낮게 측정되었다.

SBERT는 성능이 제일 좋았으며, InferSent와 Universal Sentence Encoder의 성능을 능가하였다.

SBERT가 Universal Sentence Encoder보다 성능이 떨어졌던 것은 SICK-R인데, Universal Sentence Encoder의 경우 뉴스, QnA 페이지, discussion forum과 같은 곳에서 얻은 데이터로 학습했기 때문에 SICK-R의 데이터와 유사한 측면이 있기 때문이다 (반면 SBERT의 경우 BERT를 그대로 활용하기 때문에 Wikipedia를 사용).

RoBERTa도 좋은 성능을 내었지만, SBERT와 SRoBERTa사이에는 미미한 성능차이가 있을뿐이었다.

### Supervised STS

이번엔 STSb를 지도학습으로 학습시키 결과를 살펴보자.
STSb의 경우 *caption, news, forum*에서 수집한 8,628개의 문장 쌍으로 이루어져있으며, 5,479개의 train, 1,500개의 dev, 1379개의 test로 구성되어있다.

모든 실험은 10번 random seed로 진행하여 variance의 영향력을 최소화하였다.
실험은 STSb-only와 NLI+STSb 두개로 나누어 진행하였다.

![image](https://user-images.githubusercontent.com/47516855/173166572-137f22f8-f729-4b02-966c-34271351cb8b.png){: .align-center}{: width="300"}

STSb만 학습시킨 결과보다 NLI+STSb를 학습시킨 결과가 1-2 포인트 정도의 미미한 향상이 일어났다.
그러나 BERT의 cross-encoder 구조에서는 대략 3-4%의 성능을 이끌어내었다.
이 역시 BERT와 RoBERTa의 차이는 미미하였다.

### Argument Facet Similarity






{: .align-center}{: width="600"}