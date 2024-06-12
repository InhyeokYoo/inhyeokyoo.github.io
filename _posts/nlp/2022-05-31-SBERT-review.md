---
title:  "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks review"
toc: true
toc_sticky: true
permalink: /project/nlp/review/SBERT/
categories:
  - Language Modeling
  - Semantic Textual Similarity
  - Sentence Embedding
tags:
  - NLP
  - Paper Review
use_math: true
last_modified_at: 2024-06-05
---

## Introduction

BERT가 비록 다양한 sentence classification과 sentence-pairregression task에서 SOTA를 달성했지만, 대규모의 문장 쌍을 다룰 때 연산이 많아진다는 단점이 있다.
이로인해 클러스터링이라던가, IR, semantic similarity 비교 등의 태스크에서 엄청난 오버헤드가 일어난다.

SBERT는 2019년 ACL에 accept된 논문으로, 이러한 BERT의 단점을 siamese/triplet network를 이용하여 보완한다.
논문은 [이곳](https://arxiv.org/pdf/1908.10084.pdf)에서, 코드는 [sentence-transformers 라이브러리](https://www.sbert.net/)와 [Hugging face](https://huggingface.co/sentence-transformers)에서 이용할 수 있다.

## Challenges

비록 BERTA와 RoBERTa가 Semantic Textual Similarity(STS)와 같은 문장 쌍으로 이루어진 regression task에서 SOTA를 달성하였으나, 두 개의 문장이 모델로 들어가게 되므로(cross encoder) **오버헤드가 발생**하게된다.

예를들어 BERT를 통해 $n=10000$ 문장에서 가장 유사한 두 개의 문장을 찾는다고 해보자.

이 경우 $n (n-1)/2=49995000$ (약 50M)의 inference computation을 필요로한다.
이는 V100 GPU를 사용했을 때 약 65시간정도 소요되는 양이다.
이와 비슷하게 BERT를 통해 Quora에서 40M개 이상의 질문 중 어떤 질문과 가장 비슷한지 찾는 태스크(QQP) 또한 **50시간 이상의 시간**을 소요한다.  

또한, 이러한 오버헤드 뿐만 아니라 **성능**에서도 큰 이슈가 있다.

Clustering과 semantic search에서 주로 다루는 방법은 **각 문장을 vector space로 맵핑**하여 **의미적으로 비슷한 문장은 가깝게** 만드는 식이다.
BERT의 가장 큰 단점은 문장 하나에 대해서는 임베딩을 계산할 수 없기 때문에, BERT를 사용하게 되면 주로 `[CLS]`토큰을 사용하거나, output vector의 평균을 통해 문장을 임베딩하게 된다. 
그러나 이는 좋지 않은 방법으로 문장이 임베딩되며, 심지어는 **GloVE보다도 성능이 떨어지는 것**으로 나타났다.

## Contributions

SBERT는 BERT에 siamese/triplet network를 이용, *의미적으로 의미있는(semantically meaningful)* 문장 임베딩을 가능토록한다.

여기서 *의미적으로 의미있는* 이라는 뜻은 의미적으로 **유사한 문장이 vector space 내에서 가깝다**는 것을 의미한다.
SBERT를 통해 지금까지는 적용 불가능했던 large-scale의 semantic similarity 비교, 클러스터링, semantic search를 이용한 정보 검색 등을 가능토록 한다.

10,000개의 문장 쌍에 대한 임베딩의 경우 BERT가 65시간 걸렸던 반면 SBERT는 5초 이내로 계산할 수 있게되며, 코사인 유사도의 경우 0.01초 이내로 계산할 수 있게 된다.
이는 정확히 이야기하자면 computation은 동일하지만, **sentence embedding에 특화**되어 있기 때문에 **특별한 tuning없이 문장 embedding값을 사용**하기만 하면 되기 때문인 것으로 보인다 (모델 자체의 연산감소는 없음).

성능 측면에선 STS의 경우 InferSent보다 11.7 포인트, Universal Sentence Encoder에 비해 5.5 포인트 높은 성능을 보였으며, SentEval의 경우 각 각 2.1/2.6 포인트 높은 성능을 달성하였다.

또한, 이전의 neural sentence embedding은 random initialization에서 학습을 시작하는데, SBERT의 경우 **pre-trained BERT/RoBERTa에 fine-tuning하여 학습**한다.
이를 통해 학습 시간을 상당부분 감소시킬 수 있다. 
Tuning에는 약 20분 미만이 걸렸으며, 다른 모델보다 더 좋은 결과를 얻었다.

## Related Work

대부분의 related work는 다 알거나 SBERT보다 성능이 안 좋으므로 제외하고, 가장 comparable한 poly-encoder만 보도록 하자.

![image](https://user-images.githubusercontent.com/47516855/173226990-d655bdf6-d8cf-43b6-bd8a-d8ff35f16d98.png){: .align-center}{: width="600"}

Poly-encoder는 기존의 cross-encoder(BERT)보단 연산이 빠르면서, 연산이 빠르지만 성능은 좋지 않은 Bi-eocnder(BERT 두개에 각 문장을 삽입)의 중간점에 있는 모델이다.
Cross-encoder처럼 context vector(e.g. QA의 Q, dialog의 history) **전부를 attention에 활용**하는 대신 **일부 $m$개의 vector만 추출**하여 computation은 줄이고, bi-encoder가 **단순히 dot-product**하는 것과는 달리, 미리 계산한 candidate embedding과 미리 추출한 $m$개의 vector 사이의 **attention**을 계산한다.

이를 통하여 대규모의 데이터에서 가장 유사한 문장만을 추출할 수 있었으나 score function이 symmetric하지 않다는 점과 clustering과 같이 $O(n^2)$의 연산을 필요로할 경우 overhead가 너무 크다는 단점이 있다 (연산량의 경우 SBERT는 따로 학습하지 않기 때문에 model output을 바로 뽑아서 사용한다는 측면에서 다른 점이 있는 것으로 보인다).


## Method

SBERT에선 세 가지 pooling 방법을 비교한다.
- `CLS`-token 사용
- [default] 모든 output vector에 대한 평균 (`MEAN`)
- output vector에 대한 max-over-time (`MAX`)

이 중 마지막 max-over-time은 잘 이해가 되질 않는데, [Hugging Face 코드](https://huggingface.co/sentence-transformers/nli-distilbert-base-max-pooling#usage-huggingface-transformers)를 살펴보니 각 각의 토큰 임베딩 중 가장 max인 값을 뽑아내는 것으로 보인다.

```python
# Max Pooling - Take the max value over time for every dimension. 
def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]
```

`token_embeddings`은 최상단 레이어를 의미, 여기에 padding 처리를 한 후 (`attention_mask`) `dim=1`로 주어 max값을 뽑아낸다 (`torch.max(token_embeddings, 1)[0]`).


본 세 가지 pooling에 대한 실험결과는 다음 Table에 정리되어있다 (우측상단).

![image](https://user-images.githubusercontent.com/47516855/173221157-0805334a-9518-4535-9b3f-6f831517fba1.png){: .align-center}{: width="300"}

Classification(NLI)의 경우 pooling 방법에 대한 차이가 미미하였다.
Regression(STS)의 경우 pooling strategy에 따른 성능의 차이가 컸다.
`MAX`의 경우 다른 pooling 방법보다 더 좋지 않았다.
이는 InferSent에서 BiLSTM-layer를 이용해 얻었던 결과와는 대조적이다.
따라서 `MEAN`을 기본 방법으로 사용한다.

BERT/RoBERTa를 fine-tuning하기 위해서 FaceNet의 siamese/triplet network를 사용한다.
Siamese/triplet network는 학습 데이터에 따라 달라지므로 다음 목적함수에 대해 실험을 진행한다.

### Classification Objective Function

다음은 classification(NLI)에서 사용하는 objective function이다.

![image](https://user-images.githubusercontent.com/47516855/171989665-3dfcd891-44b5-4529-aeaa-887c61a6d566.png){: .align-center}{: width="300"}

Sentence embedding $\mathbf{u}, \mathbf{v}$와 이의 element-wise difference $\lvert \mathbf{u} - \mathbf{v} \rvert$를 concatenation하고, 이를 trainable weight $\mathbf{W} _t \in \mathbb{R}^{3n \times k}$와 곱하여 objective Function을 만든다.

$$
\mathbf{o} = \text{softmax}(\mathbf{W} _t (\mathbf{u}, \mathbf{v}, \lvert \mathbf{u} - \mathbf{v} \rvert))
$$

여기서 $n$은 embedding dimension이고 $k$는 label의 갯수이다.
이후 cross entropy loss를 사용하여 이를 최적화한다.

여기서 사용한 concatenation의 경우 MT-DNN에서도 사용된바 있는데, 아래 Table에 차이에 따른 성능변화가 나타나있다.

![image](https://user-images.githubusercontent.com/47516855/173221221-3ce94c7e-a2c3-4cc2-a5b8-205c11d9407c.png){: .align-center}{: width="300"}

앞서 본 pooling strategy와는 다르게 **concatenation에 따른 성능 차이는 훨씬 큰 것**으로 나타났다.
Universal Sentence Encoder/InferSent 모두 concatenation으로 $(u, v, \lvert u - v \rvert, u \ast v)$를 사용한다.
그러나 실험 결과 $u \ast v$를 추가할 경우 성능이 하락하는 것을 볼 수 있다.

반면 $\lvert u - v \rvert$는 성능에 가장 중요한 것으로 나타났다.
이는 **두 sentence embedding 차원사이의 거리를 측정, 비슷한 문장은 가깝게하고 다른 문장은 멀리**하게 만들기 때문이다.

단, 여기서 주의할 점은 concatenation은 학습 시에만 적용될 뿐, **inference 시에는 cosine-similarity**를 통해 계산된다.

### Regression Objective Function

다음은 regression task(AFS, STS)에서 사용하는 objective function이다.

![image](https://user-images.githubusercontent.com/47516855/172142599-a77e56d8-f32d-4e1f-86b7-f93dc70e9fc8.png){: .align-center}{: width="300"}

$\mathbf{u}, \mathbf{v}$간의 코사인 유사도가 위 그림과 같이 계산된다.
이에 대한 objective function으로 MSE loss를 사용한다.

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

### Evaluation - Semantic Textual Similarity

보통 딥러닝 모델은 복잡한 regression function을 학습하여 sentence embedding과 유사도 사이의 맵핑을 가능케한다.
그러나 이러한 regression function은 문장 쌍을 필요로하며, 그러다보니 가능한 문장 쌍의 조합이 너무나 많기 때문에 scalable하기 쉽지 않다.

대신 SBERT는 cosine-similarity를 사용하여 두 문장간의 유사도를 비교한다.
다른 distance metric인 negative Manhatten/Euclidean distance도 실험하였지만 코사인 유사도와 비슷한 성능을 내었다고 한다.

### Unsupervised STS

본 실험은 STS에 학습하지 않고 테스트를 진행한 결과이다.
실험에서 사용한 데이터는 STS tasks 2012 - 2016와 SICK이다.
STS와 SICK 모두 문장 쌍 간의 의미적 유사도를 0에서 5사이로 표현한다.

피어슨 상관계수의 경우 STS에서 사용하기가 좋지 않다고 한다 ([Reimers et al., 2016](https://aclanthology.org/C16-1009/)).
따라서 문장 임베딩과 레이블간의 코사인 유사도에 대해 **스피어만 상관계수**를 통해 성능을 측정한다.

![image](https://user-images.githubusercontent.com/47516855/172149711-ccf1baeb-b02d-4125-8ae7-df7bf268fda2.png){: .align-center}{: width="700"}

BERT를 그대로 사용하는 것이 제일 안 좋았으며, **GloVe보다도 성능이 낮게** 측정되었다.

반면 SBERT는 성능이 제일 좋았으며, InferSent와 Universal Sentence Encoder의 성능을 능가하였다.

SBERT가 Universal Sentence Encoder보다 성능이 떨어졌던 것은 SICK-R인데, Universal Sentence Encoder의 경우 뉴스, QnA 페이지, discussion forum과 같은 곳에서 얻은 데이터로 학습했기 때문에 SICK-R의 데이터와 유사한 측면이 있기 때문이다 (반면 SBERT의 경우 BERT를 그대로 활용하기 때문에 Wikipedia를 사용).

RoBERTa도 좋은 성능을 내었지만 SBERT와의 성능차이가 미미했다.

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

또한, Argument Facet Similarity (AFS) corpus에 대해서도 추가적으로 실험하였다.
AFS는 논란이 많은 총기 규제, 동성혼, 사형제도에 대해 소셜 미디어에서 수집한 6천여개의 문장(sentential argument)으로 구성되어있다.
데이터는 0점(완전히 다름)부터 5점(완전히 동일) 사이의 점수가 매겨져있다.

AFS corpus에서의 유사도는 STS에서의 유사도와 많이 다른데, STS의 경우 기술적인(descriptive) 반면 AFS는 대화로부터 논증적으로 발췌한 것이 때문이다.
AFS 논문에서는 STS와의 차이점을 다음과 같이 소개하고 있다.

> We distinguish AFS from STS because: (1) our data are so different: **STS data consists of descriptive sentences whereas our sentences are argumentative excerpts** from dialogs; and (2) our definition of facet allows for sentences that express opposite stance to be realizations of the same facet (AFS = 3) in Fig. 10.

논쟁이 비슷하려면 반드시 **비슷한 주장을 할 뿐만 아니라 그 근거까지도 비슷**해야한다.
더욱이 AFS의 문장들에는 어휘의 차이가 현저하게 크기 때문에 단순한 비지도학습과 SOTA STS 모델의 성능이 매우작다.

SBERT는 다음과 같은 두 시나리오를 통해 데이터셋을 평가한다.
1. AFS를 제안한 [Misra et al. (2016)](https://aclanthology.org/W16-3636.pdf)와 같이 **10-fold cross-validation**을 사용한다. 다만 다른 토픽에 대해서 모델이 얼마나 잘 일반화되는지는 아직 명확하지 않다는 단점이 있다.
2. 1의 단점으로 인해 cross-topic setup으로 평가한다. **세 개중 두 개만 학습하고, 나머지 하나로 평가**한다. 이를 모두 반복한 후 평균을 낸다.

비록 피어슨 상관계수($p$)가 STS에 적합하진 않지만 Misra et al.과 비교하기 위해 피어슨 상관계수도 제공한다.

![image](https://user-images.githubusercontent.com/47516855/173185548-34be04f7-68c0-44ce-9efa-102040fd6468.png){: .align-center}{: width="400"}

tf-idf, average GloVe embeddings, InferSent과 같은 **비지도학습 방법의 경우 상당히 낮은 성능**을 냈다. 
SBERT를 10-fold cross-validation(1번 방법)으로 학습할 경우 BERT와 거의 동등한 성능을 냈다.

그러나 cross-topic evaluation(2번 방법)의 경우 SBERT는 Spearman correlation에서 **약 7포인트의 하락**이 일어났다.
논쟁이 비슷하려면 반드시 비슷한 주장을 할 뿐만 아니라 그 근거까지도 비슷해야하는데, BERT는 **두 문장간의 직접적인 attention(e.g. 단어끼리의 비교)이 가능**한반면, SBERT는 **단일문장을 각 각 embedding**하여 **학습 시에 없었던 토픽**에서의 문장을 비슷한 주장과 근거를 가깝게 위치하는 vector space에 맵핑해야되기 때문에 훨씬 더 어렵기 때문이라 밝히고 있다.

### Wikipedia Sections Distinction

[Dor et al. (2018)](https://aclanthology.org/P18-2009/)에서 사용된 데이터로, sentence embedding을 위한 주제별로 세밀한(thematically fine-grained) 데이터셋을 제공한다.
Wikipedia의 경우 특정 측면을 다루는 별개의 섹션이 존재하는데, Dor et al.은 같은 섹션 내 문장은 다른 섹션의 문장보다 주제별로 더 가깝다고 가정하여 대규모의 weakly labeled sentence triplet을 만들었다. 
Anchor와 positive example의 경우 같은 섹션에서, negative example은 같은 article 내 다른 섹션에서 추출한 문장으로 구성한다.

앞서 제안한 Triplet Objective를 사용하여 한 epoch 당 1.8M의 데이터셋을 학습하였으며, 222,957의 test data로 평가하였다.
학습과 테스트에서 사용한 데이터 셋은 서로 다른 Wikipedia article에서 추출하였다.
평가지표로는 정확도를 사용하여 positive example이 negative example보다 anchor에 더 가까운지 측정하였다.

![image](https://user-images.githubusercontent.com/47516855/173186119-08fce96f-f96e-4d5b-8c39-c16be72564d4.png){: .align-center}{: width="400"}

Dor et al.의 경우 BiLSTM에 triplet loss를 사용하여 fine-tuning을 진행하였다.
Table에서 확인할 수 있듯 SBERT가 Dor et al.보다 월등한 성능을 보였다.

### SentEval

SentEval은 sentence embedding의 품질을 측정하는 도구로, logistic regression classifier의 feature로 sentence embedding을 사용한다.
그 후 logistic regression classifier를 다양한 태스크에 10-fold cross-validation setup로 학습한 뒤 test-fold로 테스트를 진행한다.

그러나 SBERT의 sentence embedding은 **다른 태스크에 대한 transfer learning을 위해 개발된 것이 아니며**, 이보다는 fine-tuning BERT가 모든 레이어에 대해 업데이트를 진행하므로 새로운 태스크에 더 적합하다고 볼 수 있다.
다만 SentEval를 통해 여러 태스크에 대한 SBERT sentence embedding의 품질에 대한 인상정도는 남길 수 있을 것으로 기대하였다.

따라서 SBERT sentence embedding과 다른 embedding method를 아래와 같은 7개의 SentEval transfer tasks로 비교해보자 한다.
- **MR**: 별점 5점으로 측정된 movie reviews snippets에 대한 감정분석
- **CR**: 제품 리뷰에 대한 감정분석
- **SUBJ**: 영화 리뷰와 요약 플롯에 대한 *Subjectivity prediction*
- **MPQA**: newswire 데이터에 대한 Phrase level의 감성분석
- **SST**: Stanford Sentiment Treebank에 대한 이진분류
- **TREC**: TREC에 대한 fine grained question-type classification
- **MRPC**: Parallel news sources에서 추출한 Microsoft Research Paraphrase Corpus

*Subjectivity prediction*의 경우 처음보는 태스크이기 때문에 한번 찾아보았다.

> Thus, the target of subjectivity classifications is **to restrict unwanted and unnecessary objective texts** from further processing [(Kamal. 2013)](https://arxiv.org/ftp/arxiv/papers/1312/1312.6962.pdf).
>
> for example, although the sentence “The protagonist tries to protect her good name” contains the word “good”, **it tells us nothing about the author’s opinion** and in fact could well be embedded in a negative movie review (Pang and Lee, 2004 - SUBJ 원문 발췌).
>
> 2) classifying a sentence or a clause of the sentence as subjective or objective, and for a subjective sentence or clause classifying it as expressing a positive, negative or neutral opinion. ... The second topic goes to **individual sentences to determine whether a sentence expresses an opinion or not** (often called **subjectivity classification**), and if so, whether the opinion is positive or negative (called sentence-level sentiment classification) ([NLP-handbook](https://manoa.hawaii.edu/ccpv/workshops/NLP-handbook.pdf)).

정리하면, good이라는 단어가 포함된다고 무조건 긍정적인 리뷰가 아니기 때문에, **실제로 긍정/부정에 대한 주관적인 텍스트인지, 아니면 단순한 객관적인 텍스트인지를 구분**하는 태스크로 보인다.

아래는 이에 대한 결과이다.

![image](https://user-images.githubusercontent.com/47516855/173186979-408b4ff5-91ab-480a-ab68-023108c32544.png){: .align-center}{: width="700"}

SBERT는 7개 중 5개의 태스크에서 SOTA를 달성하였으며, InferSent/Universal Sentence Encoder보다 평균 2퍼센트 포인트 상승하였다.

SBERT sentence embedding은 감성의 정보를 잘 잡는 것으로 나타났다.
SentEval의 감성분석(MR, CR, SST) InferSent/Universal Sentence Encoder보다 **모두 큰 폭으로 향상**되었다.
SBERT가 유일하게 Universal Sentence Encoder보다 더 안 좋은 성능을 보인 것은 TREC로, 이는 QA데이터로 이루어져 있기 때문에 이로부터 이점을 얻은 것으로 보인다.
 
앞서 BERT의 평균값이나 CLS토큰의 경우 GloVe embedding의 평균보다 좋지 않은 성능을 보였는데 (Table1 참고), SentEval에서는 괜찮은 결과를 얻었으며, GloVe embedding 평균보다 좋은 성능을 보였다.
이는 STS와의 setups차이 때문이다. 

STS에서는 cosine-similarity를 사용하여 두 문장 사이의 유사도를 추정하였다.
그러나 코사인 유사도는 **모든 차원을 동일하게 처리**하는 효과가 있다.
반면 SentEval은 logistic regression classifier를 사용하고, 이로 인해 **특정 차원이 더 높거나 낮은 영향력**을 갖을 수 있게 된다.

따라서 average BERT embeddings/CLS-token의 output으로 나오는 sentence embedding은 cosine similarity, Manhatten/Euclidean distance에서 다루기 **infeasible 하다**는 결론이 나온다.

Transfer learning에서는 InferSent/Universal Sentence Encoder보다 약간 안 좋은 성능을 내었지만, siamese network를 통해 NLI를 학습하는 경우 SentEval에서 SOTA를 달성할 수 있었다.

### Computational Efficiency

이번에는 SBERT와 average GloVe embeddings, InferSent, Universal Sentence Encoder 세 개의 연산속도를 비교해본다.
본 실험에서는 STSB를 통해 실험한다.

GloVe의 경우 단순한 for-loop에 dictionary lookup을 이용, NumPy로 계산한다.
InferSent와 SBERT는 PyTorch기반으로, Universal Sentence Encoder의 경우 TensorFlow Hub에 있는 버전으로 사용한다.
실험 spec은 Intel i7-5820K CPU @ 3.30GHz, Nvidia Tesla V100 GPU, CUDA 9.2, cuDNN이다.

연산 효율을 위해 `BucketIterator`와 비슷한 느낌으로 batching(smart batching)을 진행하는데, mini-batch 내 가장 긴 문장만큼 padding한 채로 학습한다고 한다.
이로 인해 padding에서 오는 computational overhead를 줄일 수 있다고 한다.
이 부분은 잘 이해가 가질 않는데, 행렬연산에 특화된 구조를 이용하는 것으로 보인다.

아래는 이에 대한 [코드](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L517)이다.
근데 문제는 `smart_batching_collate`가 `DataLoader`의 `collate_fn`으로 들어가는데, 이 때 문장의 길이에 대한 어떠한 정보도 넘겨주지 않는다.
따라서 도대체 문장 길이와 이 smart batching의 관계를 알 수가 없다.
Tokenize도 안되어있는데 `Dataset`단에서 이를 미리 정렬해서 줄 수도 없는거고, 미리 정렬되어 있지도 않는 구조고...

```python
def smart_batching_collate(self, batch):
  """
  Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
  Here, batch is a list of tuples: [(tokens, label), ...]
  :param batch:
      a batch from a SmartBatchingDataset
  :return:
      a batch of tensors for the model
  """
  num_texts = len(batch[0].texts)
  texts = [[] for _ in range(num_texts)]
  labels = []

  for example in batch:
      for idx, text in enumerate(example.texts):
          texts[idx].append(text)

      labels.append(example.label)

  labels = torch.tensor(labels).to(self._target_device)

  sentence_features = []
  for idx in range(num_texts):
      tokenized = self.tokenize(texts[idx])
      batch_to_device(tokenized, self._target_device)
      sentence_features.append(tokenized)

  return sentence_features, labels
```

다만 또 신기한 것은 sentence embedding을 계산하는 [`encode` 함수](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L156)에서는 각 문장의 길이를 정렬하여 계산한다.
참으로 이해하기가 어려운 구조이다.

```python
...
all_embeddings = []
length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
    sentences_batch = sentences_sorted[start_index:start_index+batch_size]
    ...
```

이 부분은 sentence-transformers github [issue](https://github.com/UKPLab/sentence-transformers/issues/1592#event-6792452357)에 직접 물어보았는데, inference(`encode` method)에만 적용되고, `smart_batching_collate`에선 적용되지 않는다고 한다.
근데 이럴거면 이름을 다르게 지었어야되지 않나 싶다.

![image](https://user-images.githubusercontent.com/47516855/173226682-2d22838d-3205-4fdc-90e9-878659fb0e66.png){: .align-center}{: width="400"}

CPU만 사용할 경우 구조의 단순함으로 인해 SBERT보다 InferSent가 65% 빨랐다 (BiLSTM vs. 12 Transformer layers).
그러나 GPU를 이용할 경우 Transformer의 장점이 발휘되기 때문에 이야기가 조금 달라진다.

GPU와 앞서 설명한 batching을 이용하면 InferSent보다 약 9% 빠르며 Universal Sentence Encoder보다는 무려 55%나 빠르다고 한다.
본 batching을 CPU로 적용할 경우엔 89%, GPU에선 48%가 빠르다고 한다.

아주 당연하게도 가장 빠른 method는 average GloVe embeddings이 기록했다.

## Summary

간략하게 요약해보자.

- BERT의 sentence embedding 성능은 좋지가 않다 (GloVe보다도 낮음).
- Sentence embedding을 훨씬 더 빠르고 좋게 진행할 수 있다.
  - 다만 연산이 빨라지는 것은 아니며, 일종의 mapping을 잘 하는 모델을 만들어 처음 보는 문장이라도 좋은 representation을 얻게끔 만든다.
  - BERT와 다르게 반드시 pair로 문장이 필요로 하지 않는다.
- Task마다 다른 loss function을 사용하여 모델을 학습한다.
  - Classification의 경우 각 문장과 문장의 차이를 이용하여 softmax를, regression은 MSE를 사용한다.
- 그 결과 SOTA를 달성하였다.