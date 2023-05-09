---
title:  "[작성 중] Big Bird: Transformers for Longer Sequences review"
toc: true
toc_sticky: true
permalink: /project/nlp/Big-Bird/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2022-09-04
---

## Introduction

[A Survey of Transformers](https://arxiv.org/abs/2106.04554)

## Challenging

the full self-attention have computational and memory requirement that is quadratic in the sequence length. 
while the corpus can be large, the sequence length, which provides the context in many applications
is very limited.

![image](https://user-images.githubusercontent.com/47516855/188264775-959f7289-419a-4f3f-ba2a-586062eaeb7e.png){: .align-center}{: width="500"}



> Self-attention plays an important role in Transformer, but there are two challenges in practical applications.
> 1. Complexity. As discussion in Sec. 2.3, the complexity of self-attention is O (𝑇 2 · 𝐷). Therefore, the attention module becomes a bottleneck when dealing with long sequences.
> 2. Structural prior. Self-attention does no assume any structural bias over inputs. Even the order information is also needed to be learned from training data. Therefore, Transformer (w/o pre-training) is usually easy to overfit on small or moderate-size data.
> 
> The improvements on attention mechanism can be divided into several directions:
> 1. Sparse Attention. This line of work introduces sparsity bias into the attention mechanism, leading to reduced complexity.
> 2. Linearized Attention. This line of work disentangles the attention matrix with kernel feature maps. The attention is then computed in reversed order to achieve linear complexity.
> 3. Prototype and Memory Compression. This class of methods reduces the number of queries or key-value memory pairs to reduce the size of the attention matrix.
> 4.  Low-rank Self-Attention. This line of work capture the low-rank property of self-attention.
> 5.  Attention with Prior. The line of research explores supplementing or substituting standard attention with prior attention distributions.
> 6.  Improved Multi-Head Mechanism. The line of studies explores different alternative multi-head mechanisms.

BigBird의 경우엔 Sparse


However, while we know that self-attention and Transformers are useful, our theoretical understanding
is rudimentary. What aspects of the self-attention model are necessary for its performance? What
can we say about the expressivity of Transformers and similar models?

For example, the
self-attention does not even obey sequence order as it is permutation equivariant

permutation equivariant

Permutation equivariance는 adjacency matrix의 순서가 바뀌는대로 output의 순서도 바뀌는 함수의 특성을 말한다. 이를 식으로 표현하면 다음과 같다.

a model that does not produces the same output regardless of the order of elements in the input vector.

https://www.quora.com/What-does-it-mean-that-a-neural-network-is-invariant-to-permutation-When-does-this-happen

This concern has
been partially resolved, as Yun et al. [104] showed that transformers are expressive enough to capture
all continuous sequence to sequence functions with a compact domain. Meanwhile, Pérez et al. [72]
showed that the full transformer is Turing Complete (i.e. can simulate a full Turing machine). Two
natural questions arise: Can we achieve the empirical benefits of a fully quadratic self-attention
scheme using fewer inner-products? Do these sparse attention mechanisms preserve the expressivity
and flexibility of the original network?

Two natural questions arise: 
- Can we achieve the empirical benefits of a fully quadratic self-attention scheme using fewer inner-products?
- Do these sparse attention mechanisms preserve the expressivity and flexibility of the original network?

## Contributions

In this paper, we address both the above questions and produce a sparse attention mechanism that
improves performance on a multitude of tasks that require long contexts. We systematically develop
BIGBIRD, an attention mechanism whose complexity is linear in the number of tokens (Sec. 2). We
take inspiration from *graph sparsification* methods and understand where the proof for expressiveness
of Transformers breaks down when full-attention is relaxed to form the proposed attention pattern.
This understanding helped us develop BIGBIRD, which is theoretically as expressive and also
empirically useful. In particular, our BIGBIRD consists of three main part:

- A set of $g$ global tokens attending on all parts of the sequence.
- All tokens attending to a set of $w$ local neighboring tokens.
- All tokens attending to a set of $r$ random tokens.

This leads to a high performing attention mechanism scaling to much longer sequence lengths (8x).
To summarize, our main contributions are:

1. BIGBIRD satisfies all the known theoretical properties of full transformer (Sec. 3). In particular,
we show that adding extra tokens allows one to express all continuous sequence to sequence
functions with only O(n)-inner products. Furthermore, we show that under standard assumptions
regarding precision, BIGBIRD is Turing complete.
2. Empirically, we show that the extended context modelled by BIGBIRD benefits variety of NLP
tasks. We achieve state of the art results for question answering and document summarization on
a number of different datasets. Summary of these results are presented in Sec. 4.
3. Lastly, we introduce a novel application of attention based models where long contexts are
beneficial: extracting contextual representations of genomics sequences like DNA. With longer
masked LM pretraining, BIGBIRD improves performance on downstream tasks such as promoter-
region and chromatin profile prediction (Sec. 5)

## Related work

There have been a number of interesting attempts, that were aimed at alleviating the quadratic
dependency of Transformers, which can broadly categorized into two directions

First line of work
embraces the length limitation and develops method around it. Simplest methods in this category
just employ sliding window (Multi-passage bert), but in general most work fits in the following general paradigm:
using some other mechanism select a smaller subset of relevant contexts to feed in the transformer
and optionally iterate, i.e. call transformer block multiple time with different contexts each time.
Most prominently, SpanBERT [ 42 ], ORQA [54 ], REALM [ 34], RAG [57] have achieved strong
performance for different tasks. However, it is worth noting that these methods often require significant
engineering efforts (like back prop through large scale nearest neighbor search) and are hard to train.

Second line of work questions if full attention is essential and have tried to come up with approaches
that do not require full attention, thereby reducing the memory and computation requirements.
Prominently, Transformer-xl, [Sukhbaatar et al.](https://aclanthology.org/P19-1032/), compressive transformer have proposed auto-regresive models
that work well for left-to-right language modeling but suffer in tasks which require bidirectional
context. Child et al. [16] proposed a sparse model that reduces the complexity to O(n√n), Kitaev
et al. [49] further reduced the complexity to O(n log(n)) by using LSH to compute nearest neighbors.

Finally,
our work is closely related to and built on the work of Extended Transformers Construction [4].
This work was designed to encode structure in text for transformers. The idea of global tokens was
used extensively by them to achieve their goals. Our theoretical work can be seen as providing
a justification for the success of these models as well. It is important to note that most of the
aforementioned methods are heuristic based and empirically are not as versatile and robust as the
original transformer, i.e. the same architecture do not attain SoTA on multiple standard benchmarks.
(There is one exception of Longformer which we include in all our comparisons, see App. E.3 for a
more detailed comparison). Moreover, these approximations do not come with theoretical guarantees.

E.3 Relationship to Contemporary Work

Longformer introduced localized sliding window to reduce computation. A
more recent version, which includes localized sliding windows and global tokens was introduced
independently by Longofrmer[8]. Although BIGBIRD contains additional random tokens, there are
also differences in the way global and local tokens are realized. In particular even when there is no
random token, as used to get SoTA in question answering, there are two key differences between
Longformer and BIGBIRD-etc (see [4]):
1. We use global-local attention with relative position encodings enables it to better handle
structured inputs
2. Unlike Longformer, we train the global tokens using CPC loss and learn their use during
finetuning.

## Method

### BigBird

*Generalised attention mechanism*은 transformer 레이어 내에서 어떤 input sequence $\mathbf X = (\mathbf x _1, \cdots, \mathbf x _n \in \mathbb R ^{n \times d} $에 대해 동작하는 메커니즘을 일컫는다.
Generalized attention mechanism은 directed graph $D$로 표현되며, 이의 vertex set(그래프 내 모든 노드의 집합)은 $[n] = \{1, \cdots, n \}$로 표현된다. Arc set(directed edges의 집합)은 어텐션이 적용될 inner product의 집합으로 표현된다.

여기서 $N(i)$를 노드 $i$의 *out-neighbors(outgoing neighbors)*라 하자. Out-neighbors라 함은 어떤 노드($i$)에서 출발하는 엣지가 있는 노드들을 의미한다 (반대의 경우, 즉, 특정 노드로 향하는 엣지가 있는 경우는 *in-neighbors(incoming neighbors)*라 한다).
그러면 generalized attention mechanism의 $i^{\text{th}}$ output vector는 다음과 같이 정의된다.

$$
\text{Attn} _D (\mathbf X) _i = \mathbf x _i + \sum^H _{h=1} \sigma (Q _h (\mathbf x _i) K _h (\mathbf X _{N(i)})^\intercal) \cdot V _h (\mathbf X _{N(i)}) \tag{AT}
$$

여기서 $Q _h, K _h : \mathbb R^{d} \to \mathbb R^{m}, V _h: \mathbb R^{d} \to \mathbb R^{d}$은 각 각 query, key, value function을 의미하고, $\sigma$는 hardmax나 softmax같은 scoring function을 의미한다. $H$는 multi-head attention의 갯수를 의미한다.
또한, $X _{N(i)}$는 모든 인풋이 아닌 $\mathbf x _j : j \in N(i)$에 해당하는 인풋으로만 구성된 행렬을 의미한다.
만약 $D$가 complete digraph라면 이는 vanilla transformer가 된다.

지금까지의 notation을 간단하게 설명하면, 그래프 $D$의 adjacency matrix $A$에 적용되는 연산이며, $A \in [0, 1]^{n \times n}$이고 각 원소 $A(i, j)$는 query $i$가 key $j$에 attend하면 1이 되고 그 외의 경우엔 0이 된다.
마찬가지로 full attention의 경우 $A$의 모든 원소는 1이 되며, quadratic complexity를 갖게 된다.

이러한 self-attention을 fully connected graph로 보는 관점은 그래프 이론을 활용할 수 있게 해주고, 그래프 관련 방법론을 이용해서 복잡도를 줄일 수 있게된다.
따라서 self-attention의 quadratic complexity는 이제 *graph sparsification problem*으로 바뀌게 된다.
또한, 임의의 그래프는 expander이며, spectral property 등의 다양한 관점에서 complete graph로 근사할 수 있다.

본 어텐션을 sparse random graph로 표현하는 경우 만족해야 하는 것은 두 가지이다. 
첫번째는 노드들간 평균 이동 거리가 작아야 하고, 두번째는 locality이다.

### small average path length between node

*Erdős–Rényi model*로도 알려진 가장 단순한 random graph construction을 생각해보자.
이 경우 각 엣지는 고정된 확률로 독립되게 선택된다.
$\tilde{\Theta} (n)$개의 엣지를 갖는 그래프의 경우 두 노드 간에 가장 짧은 경로는 노드의 갯수에 logarithmic하다.
그 결과 임의의 그래프는 complete graph에 spectrally approximate하며, adjacency matrix의 두번째 eigenvalue는 첫번째 eigenvalue로부터 멀리 떨어지게 된다.

이러한 성질은 random walk에서의 *mixing time*을 빠르게 만들어주고, 이로인해 어떠한 노드 사이에도 정보가 빠르게 흐를 수 있도록 한다.
따라서 sparse attention을 제안, 각 쿼리가 임의의 $r$개의 키에 attention 하도록한다. 

> Mixing time is essentially the time it takes for the *chain* to reach or get close to the *stationary distribution*.
> 
> Definition 14.3 (Mixing Time). The mixing time of the chain corresponding to a graph $G$ is the smallest time $t$ such that for any starting distribution $x$,
> $$
> xP^t - \pi \leq 1/4
> $$

chain?
stationary distribution?


### Notion of locality

NLP와 computational biology에선 대부분의 문맥이 상당히 많은 양의 *locality of reference*를 보여주는 데이터를 갖는다.
여기서는 어떤 토큰에 대한 많은 정보가 이의 이웃한 토큰으로부터 얻어질 수 있다는 점이다.
Clark et al. [19]는 NLP의 self-attention model이 이웃한 토큰과의 inner-product가 매우 중요하다고 결론내었다.
Locality, 즉, 토큰의 근접성은 transformational-generative grammar와 같은 다양한 언어적 이론을 형성한다.

transformational grammar: a system of language analysis that recognizes the relationship among the various elements of a sentence and among the possible sentences of a language and uses processes or rules (some of which are called transformations) to express these relationships.

그래프 이론에서는 clustering coefficient가 connectivity에 대한 locality를 측정하는 방법이고, 많은 cliques나 near-cliques(subgraphs that
are almost fully interconnected)를 포함할수록 높은 값을 갖는다.

단순한 Erdős–Rényi random graphs는 높은 clustering coefficient를 갖진 않지만 small world graph로 알려진 a class of random graphs 높은 clustering coefficient를 갖는다.
Watts and Strogatz [94]의 모델은 average shortest path와 notion of locality 둘 사이에서 좋은 균형을 이루고 있기 때문에 BigBird와 유사성이 높다.
이들의 모델은 regular ring lattice를 생성한 후 $n$개의 노드가 각 방향으로 $w/2$개의 이웃들과 연결되는 식으로 생성된다.

이러한 형태의 attention을 **sliding window attention**로 정의한다.
Sliding window attention에선 width $w$를 갖는 self-attention이 $i$번째 query가 $i-w/2$부터 $i+w/2$의 key까지 attend한다.
수식으로 표현하면 다음과 같다.

$$
A(i, i-w/2:i+w/2)=1
$$

그러나 본 아이디어에 대한 sanity check로 BERT의 성능에 근접할 수 있는지 테스트해본 결과 random blocks과 local window로는 충분하지 않다는 결과가 나왔다.

![image](https://user-images.githubusercontent.com/47516855/190448873-8caf058d-fea1-45c4-a5b3-707eecda68e2.png){: .align-center}{: width="300"}

따라서 본 논문의 theoretical analysis (Sec. 3)에 기반, **global tokens**을 추가하여 성능을 향상.
이는 두 가지 방법으로 구현할 수 있는데,

- BigBird-ITC: 기존에 존재하던 토큰을 **global**하게 만듬 (internal transformer construction, ITC)
- BigBird-ETC: CLS같은 새로운 토큰을 만들어 **global**하게 만듬 (Extended Transformer Construction, ETC)

### Implementation details

GPU나 TPU같은 hardware accelerator는 연속된 byte block을 한번에 load하는 coalesced memory operation에서 유용하다. 
그러나 본 BigBird에서 동작하는 sliding window나 random element query로 인한 소량의 비규칙적인 look-up의 경우 이러한 accelerator를 활용하기가 어렵다.
본 논문에서는 look-up을 **blockifying**(블록화)하여 해결한다.

앞서 본 adjacency matrix가 sparse한 경우엔 이는 GPU에서 효율적으로 구현하기가 어렵다.
GPU는 몇천개의 코어를 통해 연산을 병렬적으로 수행하기 때문이다.

이를 위하여 attention pattern을 블록화한다.
즉, query와 key를 함께 packing한 후, 이 블록에 대해 attention을 정의한다.

그림과 같이 12개의 query/key vector가 있고, block size를 2라 하자.
그러면 query/key matrix를 $12/2=6$개의 block으로 나눌 수 있다.
이렇게 완성된 block matrix에 대해 앞서 살펴본 attention pattern을 적용한다.

1. Random attention: 각 query block이 $r$개의 임의의 key block으로 attend한다. 그림의 경우 $r=1$이 된다.
2. Window local attention: query/key block의 수가 같아야 block window를 진행할 수 있다. 모든 $j$번째 query block이 $j-(w-1)/2$부터 $j+(w-1)/2$ block까지 attend한다. 그림의 경우 $w=3$이 된다.

Turing Complete(튜링 완전)
튜링 완전(turing complete)이란 어떤 프로그래밍 언어나 추상 머신이 튜링 머신과 동일한 계산 능력을 가진다는 의미이며 튜링 머신으로 풀 수 있는 문제, 즉 계산적인 문제를 그 프로그래밍 언어나 추상 머신으로 풀 수 있다는 의미.

> The point of stating that a mathematical model is Turing Complete is to reveal the capability of the model to perform any calculation, given a sufficient amount of resources (i.e. infinite), not to show whether a specific implementation of a model does have those resources. Non-Turing complete models would not be able to handle a specific set of calculations, even with enough resources, something that reveals a difference in the way the two models operate, even when they have limited resources. Of course, to prove this property, you have to do have to assume that the models are able to use an infinite amount of resources, but this property of a model is relevant even when resources are limited. [출처](https://stackoverflow.com/questions/2990277/how-useful-is-turing-completeness-are-neural-nets-turing-complete)

Pérez et al. [72] showed that the full transformer based on a quadratic attention
mechanism is Turing Complete. This result makes one unrealistic assumption, which is that the
model works on arbitrary precision model. Of course, this is necessary as otherwise, Transformers
are bounded finite state machines and cannot be Turing Complete. -> ????




## Summary

{: .align-center}{: width="300"}
