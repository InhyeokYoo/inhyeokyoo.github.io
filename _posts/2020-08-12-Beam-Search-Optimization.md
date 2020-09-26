---
title:  "Beam Search Optimization (미완성)"
excerpt: "Beam Search에 대한 이해와 구현"
toc: true
toc_sticky: true

categories:
  - PyTorch
  - NLP

use_math: true
last_modified_at: 2020-08-12
---

# Introduction

[Transformer 구현](https://inhyeokyoo.github.io/pytorch/nlp/NLP-Transformer-Impl-Issues/)을 하던 도중 Beam Search에 대한 이해가 부족한 것 같아 정리를 해보려 한다.
논문의 링크는 [여기](https://arxiv.org/pdf/1606.02960.pdf)를, reference는 따로 밑에다 달도록 하겠다.

# Introductuion & Related Work

- seq2seq을 학습하는 주된 방법은 conditional language model
    - input sequence와 target words의 *gold* history를 조건부로 각 연속적인 target word의 likelihood를 maximizing
- 따라서 학습은 word-level의 loss를 계산하고, 이는 주로 target vocabulary에 대한 cross entropy loss가 됨
- 그러나 seq2seq은 test-time에서 conditional language model로 사용되지 않음
    - 대신 반드시 온전한 형태의 word sequence를 생성해야 함
- 실제로는 대부분 beam search나 greedy search를 사용해서 단어를 생성
- 이러한 맥락에서 [Ranzato et al. (2016)](https://arxiv.org/abs/1511.06732)는 학습과 생성의 구조 조합이 다음과 같은 두 가지 major한 문제를 야기
    - *Exposure Bias*: 모델은 학습과정에서 training data distribution에만 노출되고, 자신이 직접 생성한 데이터에는 노출되지 않아서 문제가 발생
    - *Loss-Evaluation  Mismatch*: 학습에서 사용된 loss-function은 word-level에서 적용됨. 그러나 모델의 성능은 discrete metric (e.g. BLEU)을 통해 평가
        - BLEU를 학습에서 안 쓰는 이유는 differentiable하지 않고,
        - optimization을 조합하여 쓰는 것은 given context에 대해 어떠한 sub-string을 maximize할 것인지 결정해야 하기 때문
    - 이외에도 *label bias* ([Lafferty et al.,  2001](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers))문제를 고려
        - Label bias는 [이곳](https://www.quantumdl.com/entry/Endtoend-Sequence-Labeling-via-Bidirectional-LSTMCNNsCRF)을 참조
- 

# Background and Notation

- 일단 input sequence가 인코딩되면, seq2seq은 *디코더*를 사용하여 target vocabulary $\mathcal V$로부터 target sequence를 생성
    - 특히, 이 생성되는 단어들은 input representation $x$와 이전에 생성된 단어들이나 *history*에 조건부로 생성
- 여기서 우리는 $w_{1:T}$라는 notation을 사용하여 T길이의 임의의 sequence를 표현할 것임
- $y_{1:T}$는 x에 대한 *gold*(정답) target word sequence를 의미함
- $m_1, ..., m_T$를 vector T개의 sequence, $h_0$를 initial state vector로 하자
- 이러면 RNN에 어떠한 sequence를 적용하더라도 다음과 같은 $h_t$를 생성한다
    - $h_t \leftarrow \textrm{RNN}(m_t, h_{t-1}; \theta)$
    - 여기서 $m_t$는 항상 target word sequence $w_{1:T}$에 대응하는 embedding vector
    - 따라서 다음과 같이 써도 무방함
        - $h_t \leftarrow \textrm{RNN}(w_t, h_{t-1}; \theta)$
- 그리고 $p(w_t \rvert w_{1:t-1}, x) = g(w_t, h_{t-1})$를 통해서 conditional laguage modeling을 학습함
    - 이는 $x$와 target history($w_{1:t-1}$)에 조건부로 t번째 target word의 확률을 model하는 것
    - $g$는 보통 affine layer와 softmax를 의미
- 완성된 모델은 neural language model과 유사하게 매 time step에서 gold history에 조건부로 하며 cross-entropy loss를 minimize
    - $- \ln \Pi^T_{t=1} p(y_t \rvert y_{1:t-1}, x)$
- 디코더가 학습되고 나면, discrete sequence의 생성은 conditional distribution $\hat y_{1:T} = \mathrm{argbeam_{w _{1:T}} } \Pi^T _{t=1} p(w_t \rvert w _{1:t-1}, x)$에 따라 target sequence의 확률을 maximizing하여 실행된다
    - 여기서 notation $\textrm{argbeam}$는 디코딩 프로세스에 휴리스틱한 search가 필요하기 때문에 강조
    - 디코더는 non-Markovian이기 때문
- 간단한 beam search 프로시져는 나중의 K개의 history를 탐색하는데, 이는 디코더에 매우 효과적으로 알려져 있음
- 그러나, 위에서도 언급했듯, conditional language-model style training 이후에 이런방식으로 decoding하는 것은 *잠재적으로* exposure bias와 label bias문제를 겪을 가능성이 있다.

# Beam Search Optimization (BSO)

# Conclusion

결국 Transformer에서 BSO를 쓰지는 않는 것으로... RNN계열에서 사용되는 것인데, 현재 RNN은 거의 사용하지 않으므로 크게 신경쓸 필요가 없어보인다.
따라서 포스트 중도포기.