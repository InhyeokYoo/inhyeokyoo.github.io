---
title:  "[작성 중] ETC: Encoding Long and Structured Inputs in Transformers review"
toc: true
toc_sticky: true
permalink: /project/nlp/ETC/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2022-11-06
---

## Introduction

논문: https://aclanthology.org/2020.emnlp-main.19.pdf
깃헙: https://github.com/google-research/google-research/tree/master/etcmodel
papertalk: https://papertalk.org/papertalks/11815


## Challenging

(1) scaling input length: The computational and memory complexity of
attention in the original Transformer scales quadrat-
ically with the input length, typically limiting input
length to around 512 tokens.

Consider question answering (QA) tasks
that require reasoning across multiple documents
(e.g., the HotpotQA dataset (Yang et al., 2018)) all
of which must simultaneously fit in the model input.

Many approaches have been proposed
to address this, like hierarchical processing (Zhang
et al., 2019), sparse attention (Child et al., 2019),
and segment-level recurrence (Dai et al., 2019).

(2) encoding structured inputs: few models focus on
structured inputs, by which we refer to any underly-
ing graph or hierarchical structure among the input
tokens.





## Contributions

To address these challenges, we present a novel
attention mechanism called global-local attention,
which divides the input into two sequences (which
we call the global input and the long input).

This
mechanism introduces local sparsity to reduce
the quadratic scaling of the attention mechanism.
When this is coupled with relative position encod-
ings (Shaw et al., 2018), it allows for handling
structured inputs in a natural way

Additionally,
unlike previous Transformer extensions, ETC can
be initialized from existing pre-trained standard
BERT models (which together with a GPU/TPU-
friendly implementation, allows for efficient model
training) (An exception to this is Longformer (Beltagy et al., 2020),)
Our results show that initializing from RoBERTa (Liu et al., 2019) significantly improves
performance.

Finally, we show that by adding a
pre-training Contrastive Predictive Coding (CPC)
task (Oord et al., 2018), performance improves
even further for tasks where structure is important,
as CPC plays the role of a masked language model
(MLM) task, but at a sentence level of granularity.

In this
paper, we consider only the encoder side of the
Transformer, and leave the decoder for future work.

## Related work

Many variants of the original Transformer
model (Vaswani et al., 2017)  typically limit inputs to n = 512 to-
kens due to the O(n2) cost of attention.
We classify
prior approaches to scale up attention into four cat-
egories: sparse attention, recurrence, hierarchical
mechanisms, and compressed attention.

### Sparse Attention

limiting each token
to attend to a subset of the other tokens

the Sparse Transformer (Child et al., 2019)
used predefined attention patterns for both text and
image generation.
They showed that attending only to previous pixels in the same row or column was
enough to generate high quality images, while keep-
ing attention cost at O(n√n).

In the Adaptive Attention Span Transformer (Sukhbaatar et al., 2019)
each attention head is associated with a decaying
learnable masking function, which limits the number of tokens it can attend to.
They show that lower
layers learn to use short attention spans, and only
in higher layers are attention spans longer.

Sparse
attention has also been used to increase the inter-
pretability of attention heads by allowing attention
to assign exactly zero weight to certain input to-
kens (Correia et al., 2019).

The Reformer (Kitaev
et al., 2020) model finds the nearest neighbors of
the attention query (those input tokens that would
result in the highest attention weights) using local-
ity sensing hashing (Andoni et al., 2015) and only
uses those for attention. This reduces attention cost
to O(n log(n)).

The Routing Transformer (Roy
et al., 2020) learns dynamic sparse attention pat-
terns using online k-means, reducing complexity to
O(n1.5).

Finally, the most related approach to the
work presented in this paper is Longformer (Belt-
agy et al., 2020), developed concurrently to ETC,
and which features a very similar global-local at-
tention mechanism as ETC’s but does not directly
encode graph or hierarchical structure (more de-
tailed comparison in Section 3).

### Recurrence

Recurrence incorporates elements of recur-
rent neural networks into Transformer models to
lengthen their attention span.

Transformer-XL (Dai
et al., 2019) takes this approach, dividing the input
sequence into segments and then processing these
segments one at a time

### Hierarchical Mechanisms

the input sequence is split into blocks that are ingested inde-
pendently to produce summary embeddings that
represent the whole block. Then, separate layers
ingest the concatenation of these embeddings

HIBERT (Zhang et al., 2019) uses this
idea at the sentence level for extractive summariza-
tion (illustrated in the bottom-left of Figure 1)

Hi-
erarchical attention in Transformers has also been
applied to other NLP tasks such as neural machine
translation (Maruf et al., 2019).

Moreover, notice
that this idea of processing the input hierarchically is not specific to Transformer models, and it has
been applied to recurrent neural network models
both at the level of sentences (Yang et al., 2016;
Miculicich et al., 2018) and blocks (Shen et al.,
2018).

### Compressed Attention

takes the idea of hier-
archical attention one step further by selectively
compressing certain parts of the input

The BP-Transformer (Ye et al., 2019) model builds a binary
partitioning tree over the input, and only lets the
model attend to the leaves (the raw tokens) for
nearby tokens, and higher nodes in the tree (sum-
maries of groups of tokens) as tokens grow more
distant (see Figure 1, middle top).

memory compressed attention (Liu et al.,
2018) where groups of k tokens are compressed
via a convolution filter before they are attended
to,

Star Transformer (Guo et al., 2019),
where each token can attend only to its immedi-
ate left/right neighbors and to a separate special
auxiliary token that represents a summary of the
whole input (see Figure 1, left)

The Compressive
Transformer (Rae et al., 2019) integrates this idea
into Transformer-XL by compressing tokens in the
input that are far away. The model benefits from
detailed attention to nearby tokens, while using
summarized information for more distant tokens
(see Figure 1, lower right).

### Compared with Longformer

Longformer의 경우 하나의 문장과 이에 대한 global token이 있다는 점을 제외하면 ETC와 비슷한 attention 메커니즘을 사용한다.
주요 차이점은 다음과 같다.

1. ETC는 global-local attention, relative position encoding, flexible masking을 통해 graph neural network처럼 structured inputs을 인코딩 한다는 점.
2. Longformer의 global token은 ETC의 CPC loss처럼 절대로 pre-trained되지 않으며, 따라서 fine-tuning시에 학습해야한다.


## Method: Extended Transformer Construction

기존의 Transformer와의 중요한 차이점은 ETC는 **structed input**을 다룬다는 점이다.
이는 relative position encoding과 global-local attention, CPC pre-training을 통해 해결할 수 있다.
이를 하나씩 살펴보도록 하자.

### Relative Position Encoding

ETC에선 [Shaw et al. (2018)](https://arxiv.org/abs/1803.02155)의 연구에 영감을 받아 기존의 absolute position encoding 대신 토큰들 사이의 상대적 위치를 파악하는 relative position encoding을 사용한다.

인풋 시퀀스 $x = (x _1, \dotsc, x _n)$가 있을 때, 이는 레이블이 있는(labeld) 완전 유향 그래프 (fully connected and directed graph)로 볼 수 있다.
이 때 $l _{ij}$는 $x _i$와 $x _j$를 연결하는 edge로 볼 수 있다.
또한, maximum clipping distance $k$가 주어졌을 때, Shaw et al. (2018)은 $2k + 1$개의 *relative position labels* $l _{-k}, \dotsc, l _k$을 정의하였다.
그리고 두 인풋 토큰 사이에 있는 edge의 label은 이들의 거리 $j - i$에 의해서만 결정된다.
만일 이 둘 사이의 거리가 $k$보다 큰 경우 $l _k$로, 작은 경우 $l _{-k}$로 고정된다.
그리고 각각의 label은 learnable parameter $a^K _{l}$가 되어 attention 계산하는데 영향을 미치게 된다.

Relative position encoding은 인풋 길이와 독립적이므로 사전학습에서 나타났던 문장 길이보다 더 긴 문장이 들어와도 쉽게 적응 가능하다.
다른 최근 연구인 [Shaw et al., 2019](https://arxiv.org/abs/1905.08407)처럼 ETC의 attention은 단순히 상대적 위치 뿐만 아니라 구조화된 인풋에 유용한 임의의 토큰 사이의 관계를 표현하는데 relative position encoding을 사용한다.

--> g2g, g2l, l2g, l2l에 따라 label이 달라지는가? 수식을 봐서는 달라지지 않는 것으로 보인다.

### Global-Local Attention

ETC는 global input($x^g = (x^g _1, \dotsc, x^g _{n _g})$)과 long input($x^l = (x^l _1, \dotsc, x^l _{n _l})$) 두 개의 시퀀스를 입력받는다.
일반적으로 long input sequence에는 일반적인 Transformer 인풋보다 더 많은 인풋을 포함하는 반면, global input에는 훨씬 작은 양의 auxiliary token을 포함한다 ($n _g \ll n _l$).
그리고 attention은 global-to-global (g2g), global-to-long (g2l), long-to-global (l2g), long-to-long (l2l) 네 개의 종류로 나눠지게 된다.

가장 computationally expensive한 **l2l piece**의 attention은 고정된 radius $r \ll n _l$로 제한된다.
이렇게 길이보다 작은 radius로 제한되는 attention span으로 인한 단점을 보완하기 위해 global input내 토큰은 제한하지 않는다.
따라서 global input 토큰을 통해 radius 밖의 long input 토큰끼리 정보를 교환할 수 있다.
이로인해 g2g, g2l, l2g 토큰들끼리는 제한되지 않는다.

이러한 매커니즘은 아래의 그림에서 확인할 수 있다.

![image](https://user-images.githubusercontent.com/47516855/200129740-fd725eb0-d764-40c7-89a2-5c3d8c4d8548.png){: .align-center}{: height="500"}

ETC에서의 시간복잡도는 $O(n _g (n _g + n _l) + n _l(n _g + 2r + 1))$이며, $n _g = O(2r + 1)$로 가정하면 attention의 복잡도가 long input에 대해 linear하게 변하는 것을 확인할 수 있다 ($O(n^2 _g + n _g n _l)$)

--> 어케 계산함?

또한, per-instance Boolean attention matrix $M^{g2g}, M^{g2l}, M^{l2g}, M^{l2l}$를 통해 attention이 통하지 않아야 하는 곳(e.g. l2l)은 0으로 만들어 attention을 계산하지 않도록 한다.

각각의 g2g attention head는 다음과 같이 계산된다.
global input $x^g = (x^g _1, \dotsc, x^g _{n _g}), x^g _i \in \mathbb R^{d _x}$에 대해 attention output $z^g= (z^g _1, \dotsc, z^g _{n _g}), z^g _i \in \mathbb R^{d _z}$는 다음과 같이 계산된다.

$$
\begin{align}
z^g _i &= \sum^{n _g} _{j=1} \alpha^{g2g} _{ij} x^g _j W^V \\
\alpha^{g2g} _{ij} &= \frac{\exp{e^{g2g} _{ij}}}{\sum^n _{\ell = 1}\exp{e^{g2g} _{i \ell}}} \\
e^{g2g} _{ij} &= \frac{x^g _i W^Q (x^g _j W^K + a^K _{ij})^\intercal}{\sqrt{d _z}} - (1 - M^{g2g} _{ij}) C
\end{align}
$$

여기서 $M^{g2g}$는 앞서 말했듯 binary attention mask가 되며, $W^Q, W^K, W^V$는 learnable weight matrix, $a^K _{ij}$는 learnable vectors로 relative position label을 의미하며, $C$는 상수값으로 큰 값을 나타낸다 (실험 시 $C=10000$)

--> 그냥 마스킹하는게 아니라 왜이렇게 큰값으로 해주지? 이러면 어떻게 됨?

--> same convention as BERT  뭔 말임?

g2g 뿐만 아니라 나머지 세 개의 attention의 경우도 이와 유사하게 계산된다.
$W^K, W^V$에 대해서는 네 개의 attention에서 공유하거나 각자 학습하는 방식으로 실험한다.


### Long Inputs and Global-Local Attention

ETC에서는 long input을 적절한 segment 단위로 나누고, 이 사이사이에 global input 토큰을 넣는 방식으로 인풋을 구성한다.
그 후 relative position label을 이용하여 global segment tokens과 이에 <span style="color:#6495ED">속하는</span>/<span style="color:#FF1493">속하지 않는</span> word piece tokens을 연결해준다 (그림과 글의 색상 참고).

이에 한 가지 더해 g2l attention의 경우 **hard masking**을 통해 global token이 다른 문장의 token에 attention하는 것을 막는다.
이는 일종의 inductive bias로, global token이 이에 해당하는 **문장을 요약하도록 하는** 장치이다.
이를 통해 특정 데이터셋에선 성능이 향상하는 것을 확인하였다.

![image](https://user-images.githubusercontent.com/47516855/200162889-be9d2af4-4c0b-48b2-ac9a-796fba77bf5c.png){: .align-center}{: width="400"}

위 그림의 a는 이러한 asymmetric hard-masking(g2l)을 보여주고 있다.
여기서 색이 다른 것은 다른 relative position label을 의미한다.

그림에서 long input 토큰이 radius 근처에 있는 토큰에만 attention하지만, global token을 통해 간접적으로 다른 토큰에 attention할 수 있음을 확인할 수 있다.

### Structured Inputs

ETC에선 인풋 시퀀스 $x = (x _1, \dotsc, x _n)$에 대해, $x$ 내 토큰 사이에 존재하는 관계를 *구조(structure)*라는 용어라 칭한다.
$x$가 평문이라면 이들 사이에 존재하는 관계는 오직 토큰의 *순서*말고는 없으며, 이는 BERT가 유일하게 포착할 수 있는 구조이기도 하다.
그리고 *structured inputs*은 이러한 토큰 순서 이외의 추가적인 정보로 정의하며, ETC는 앞서 살펴본 global-local attention과 relative position label을 활용하여 structured inputs을 인코딩한다.

ETC는 특히 hierarchical structure를 잡는데 잘 어울리며, 그 이유는 다음과 같다.

1. Relative position labels는 토큰의 상대적 위치를 표현하는데 사용하였지만, Transformer를 graph neural network로 바라볼 경우 label은 *is-a*, *part-of*와 같은 관계로 확장할 수 있다.
2. Long/global input의 분리가 global input이 long input을 요약하는 자연스러운 구조를 만들게된다.
3. 두 토큰 사이에 edge를 막아야 되는 일이 생기면 $M^{g2g}, M^{g2l}, M^{l2g}, M^{l2l}$를 통해 처리할 수 있다.

![image](https://user-images.githubusercontent.com/47516855/200162902-33ec8245-2e61-4fdc-b247-a8191a3f2477.png){: .align-center}{: width="400"}

위 그림 Figure 3b에는 이러한 구조가 잘 나타나있다.
Masking과 relative position label을 이용하여 문장 내 within-context order는 포함하되 context 사이의 순서는 없는 context-sentence-token hierarchy를 표현하고 있다.

--> ??

아래는 [구글 research blog](https://ai.googleblog.com/2021/03/constructing-transformers-for-longer.html)에서 가져온 figure인데, 이를 직관적으로 잘 설명하는 것 같다.

![](https://1.bp.blogspot.com/-Byc8ml0X7E8/YFzbxUfSSpI/AAAAAAAAHXA/YOjKMYV2eN83QcsqOxMZi8DupgdlEkGmwCLcBGAsYHQ/s1280/image2.gif){: .align-center}{: width="600"}

### Pre-training Tasks

Pre-training에선 whole word masking(단어의 일부 word piece가 마스킹되는 경우 그 단어에 속하는 다른 단어도 마스킹)을 이용한 MLM과 Contrastive Predictive Coding (CPC)을 사용한다.

CPC는 latent space 내 subsequent input(i.e. 토큰 블록에 대한 내부 hidden representations)을 예측한다.
ETC는 이러한 아이디어를 차용한다., global input 문장의 요약 토큰을 이용한다.

n개의 문장에 대해 문장 내 토큰을 마스킹한다 (단, global input의 sentence summary token은 건들지 않는다).
그 후 모델이 마스킹된 문장의 global sentence summary tokens에 대한 hidden representation과 마스크되지 않은 문장에 대한 global summary token의 hidden representation간의 차이를 최소화하도록 학습한다.
Loss의 경우엔 CPC를 따라 Noise Contrastive Estimation (NCE) loss를 사용한다.

--> 뭔말인지 잘...


### Lifting Weights from Existing Models

BigBird와 비슷하게 warm start를 위해 BERT 파라미터를 가져온다.
Global-local attention는 인풋이 작거나 radius가 sparsity를 제거할만큼 클 경우 BERT의 동작방식과 유사하기 때문에 호환이 가능하기 때문이다.
다만 그럼에도 불구하고 global token과 relative positional encoding으로 인해 여전히 pre-training은 필요하다.

ETC에선 long/global에 따라 $W^Q, W^v, W^V$가 다르지만 둘 모두 BERT/RoBERTa를 이용하여 초기화한다.
그 외 feed forward layer와 embedding layer도 가져오며, NSP와 absolute position embedding은 버린다.
나머지 CPC loss와 relattive position encoding 관련된 파라미터는 임의로 초기화한다.

## Experiment

ETC의 main contribution인 long inputs과 structured input을 집중적으로 살펴보자.
아래는 실험에서 쓰인 데이터셋 정보이다.

![image](https://user-images.githubusercontent.com/47516855/200173347-43d52601-4595-4b27-9a91-21e380b22973.png){: .align-center}{: width="300"}

### NQ(Natural Questions)

구글에서 공개한 데이터 셋으로 위키피디아 article 하나와 이에 대한 질문으로 이루어져있다.
목표는 아티클 내 몇 단어로 이루어진 *짧은 답변* 과 문단(paragraph)전체로 이루어진 *긴 답변* 을 찾는 것이다.
답은 존재하지 않을 수도 있다.
성능은 사람이 만든 정답셋과 모델 결과에 대해 F1 score를 계산하여 평가한다.

ETC의 global input에는 CLS token, 질문 내 토큰 당 하나의 "question" token이 존재하며, long input에는 단락(long answer candidate) 당 하나의 "segment" token으로 구성된다.

Fine-tuning시 hyperparameter sweep은 아래에 값들로 수행한다.
- learning rates



**HotpotQA**

여러개의 context에서 evidence를 잘 조합하여 질문에 대한 답을 하는 데이터셋이다.
실험에서는 HotpotQA의 distractor setting을 사용했는데, 이에는 10개의 문단이 주어지고 이 중 2개만 유용한 정보를 포함하며 나머지는 distractor로 이루어져있다.
태스크는 질문에 대한 답변은 물론 문장단위(sentence granularity)의 evidence도 검증한다.

**WikiHop**

HotpotQA와 비슷한 형태로, 복수개의 context가 위키피디아 복수개의 article 일부에 해당한다.
목표는 특정 객체(entity)에 대한 성질을 찾는 것인데, article에는 이에 대한 설명이 주어지지 않는다.
데이터는 질의, 정답 후보군(multi-hop), 정답 후보에 대한 context로 이루어져있다.

**OpenKP**

keyphrase extraction 데이터셋으로, 문서 내 핵심문구를 추출하는데 그 목표가 있다.
OpenKP의 각 문서에는 최대 3개의 짧은 핵심문구(keyphrases)가 포함되어 있으며, 이를 예측해야한다.
OpenKP는 평문으로 이루어진(flat text sequences) 문장이 아니라 DOM 요소 간 계층적/공간적 관계, 시각적 속성을 포함하는 웹사이트이기 때문에 ETC의 structured inputs을 평가하는데 유용하다.

### Training Configuration

Base와 Large 두 세팅으로 비교한다.

|                                        |    Base    |     Large    |
|----------------------------------------|:----------:|:------------:|
| # layers                               |     12     |      24      |
| Hidden Size                            |     768    |     1024     |
| # heads                                |     12     |      16      |
| Local Attention Radius ($r$)           |     84     |      169     |
| Relative Position Maximum Distance ($k$) |     12     |      24      |

**Pre-training**의 경우 각 문장 당 하나의 auxiliary token을 global input에 넣는다.
사전은 BERT에서 사용한 30k짜리 영문 uncased word piece를 가져와서 사용한다.
데이터셋 또한 BERT와 동일하게 사용하되, 7문장 이하의 문서는 제외한다.
명시하지 않는한 base model은 학습하는데 BERT와 동일한 토큰 수를 사용하였으며, large모델은 이의 두배를 사용하여 학습한다.
또한, LAMB optimizer와 $\sqrt(8) \times 10^{-3}$의 learning rate를 사용한다.

**Fine-tuning**시에는 


**NQ**
28, 230 and 460 global
tokens for models with 512, 4096 and 8192 long

## Model Computational Requirements

### Memory

ETC에서 실험한 인풋 길이보다 더 긴 경우에서의 headroom (the maximum distance overhead (the difference between the structure gauge and the loading gauge)을 측정하기 위해 TPU v3로 gradient checkpointing와 Adam/LAMB과 같은 optimizer에서 요구하는 extra gradient moments을 이용하여 추가로 실험해보았다.

글로벌 인풋 길이는 512 토큰으로 고정하여 *base* 모델의 경우 롱 인풋의 경우 22656까지, *large* 모델의 경우 8448까지 늘리는 것이 가능하였다.

### Compute

ETC에서의 시간복잡도는 $O(n _g (n _g + n _l) + n _l(n _g + 2r + 1))$이며, $n _g = O(2r + 1)$로 가정하면 attention의 복잡도가 long input에 대해 linear하게 변하는 것을 확인할 수 있다 ($O(n^2 _g + n _g n _l)$)

![image](https://user-images.githubusercontent.com/47516855/204081134-d7e1dff5-e126-4c90-bdd2-351d01c300e6.png){: .align-center}{: width="300"}

![image](https://user-images.githubusercontent.com/47516855/204081629-e8f44fb2-8b9d-48e1-84fd-442709dda7bf.png){: .align-center}{: width="300"}

Table 6과 Table 7은 pre-training 실험에 대한 결과이다.
당연하게도 pre-training은 고도의 연산을 필요로 하므로 더 많은 양의 하드웨어를 사용하였다.
GPU를 이용한 모델을 학습하는 일반적인 사용 사례에 대해서도 인사이트를 얻기 위해 아래와 같이 NVIDIA Tesla V100을 이용하여 wall time을 비교하였다.

![image](https://user-images.githubusercontent.com/47516855/204081734-2b67eb8d-fb94-4156-b767-9100bee48392.png){: .align-center}{: width="300"}

처음엔 ETC가 느리지만 인풋 길이가 1500을 넘어가게 되면 점점 더 빨라지는 것을 볼 수 있다.
또한, BERT의 경우 메모리 제한 때문에 길이가 제한된다.
ETC는 linear하지 않는데, 이는 글로벌 인풋을 늘릴수록 롱 인풋을 늘리기 때문이다.

### Parameters

![image](https://user-images.githubusercontent.com/47516855/204082204-e2b04465-cf88-47e8-b743-39084411558a.png){: .align-center}{: width="300"}

위 표는 ETC configuration에 따른 파라미터의 수이다.
여기서 고려할 중요한 사항은 파라미터의 수가 인풋의 길이에 종속되지 않으며, 임베딩차원 $d$, layer 수 $l$, relative position labels의 수 $k$에 따라 달라진다는 것이다.

## Structured Input Example

![image](https://user-images.githubusercontent.com/47516855/200162902-33ec8245-2e61-4fdc-b247-a8191a3f2477.png){: .align-center}{: width="400"}

위 그림 Figure 3b는 WikiHop과 같이 인풋이 컨텍스트, 문장으로 이루어진 경우에 나타날 수 있는 attention 패턴이다.
이러한 컨텍스트 사이에는 순서가 없으나 컨텍스트 안에 있는 문장에는 순서가 존재한다.

![image](https://user-images.githubusercontent.com/47516855/204082429-9ca891bf-528c-454c-b4f8-6b63f50528a7.png){: .align-center}{: width="400"}

위 Figure 5는 이를 ETC에서 인코딩하는 방식을 나타낸 것이다.




## Summary

{: .align-center}{: width="300"}

he maximum distance overhead

gradient checkpointing: https://spell.ml/blog/gradient-checkpointing-pytorch-YGypLBAAACEAefHs
